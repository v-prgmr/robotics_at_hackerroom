"""Convert Orbit intermediate episodes to Stable World Model Lance format."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd

from lewm_orbit_bridge.common import (
    parse_dataset_specs,
    read_json,
    stratified_episode_split,
    write_json,
)


REQUIRED_COLUMNS = (
    "timestep_index",
    "monotonic_timestamp_s",
    "left_commanded_action",
    "right_commanded_action",
    "left_action_names",
    "right_action_names",
    "metadata_json",
    "overhead_frame_age_s",
    "overhead_video_frame_index",
    "overhead_camera_frame_index",
)


class SequentialVideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open overhead video: {path}")
        self.index = -1
        self.frame: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray:
        if index < self.index:
            raise ValueError(f"Video frame references decrease in {self.path}: {index} after {self.index}")
        if index == self.index and self.frame is not None:
            return self.frame
        while self.index < index:
            ok, bgr = self.capture.read()
            if not ok:
                raise ValueError(f"Could not decode video frame {index} from {self.path}")
            self.index += 1
            self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert self.frame is not None
        return self.frame

    def close(self) -> None:
        self.capture.release()


def git_sha(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def discover_episodes(dataset_specs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    episodes = []
    seen_uids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for collection_type, root in dataset_specs:
        episode_paths = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        if not episode_paths:
            raise ValueError(f"No episodes found in {root}")
        source_dataset = root.name
        for path in episode_paths:
            metadata_path = path / "episode_metadata.json"
            timesteps_path = path / "timesteps.parquet"
            video_path = path / "videos/overhead.mp4"
            if not all(p.is_file() for p in (metadata_path, timesteps_path, video_path)):
                raise ValueError(f"Incomplete Orbit episode: {path}")
            metadata = read_json(metadata_path)
            source_episode_id = str(metadata.get("episode_id", path.name))
            source_key = (str(root), source_episode_id)
            if source_key in seen_sources:
                raise ValueError(f"Source episode was supplied more than once: {root}/{source_episode_id}")
            seen_sources.add(source_key)
            suffix = hashlib.sha256(str(root).encode()).hexdigest()[:8]
            uid = f"{source_dataset}-{suffix}/{source_episode_id}"
            if uid in seen_uids:
                raise ValueError(f"Could not construct a unique episode UID for {path}")
            seen_uids.add(uid)
            terminal_known = collection_type in {"policy_success", "policy_failure"} and isinstance(
                metadata.get("terminal_success"), bool
            )
            episodes.append(
                {
                    "episode_uid": uid,
                    "source_dataset": source_dataset,
                    "source_dataset_path": str(root),
                    "source_episode_id": source_episode_id,
                    "collection_type": collection_type,
                    "terminal_success_known": terminal_known,
                    "terminal_success": bool(metadata.get("terminal_success")) if terminal_known else False,
                    "goal_eligible": collection_type == "expert"
                    or (terminal_known and bool(metadata["terminal_success"])),
                    "path": path,
                    "metadata": metadata,
                }
            )
    return episodes


def _as_int_array(series: pd.Series, label: str, episode_path: Path) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"Missing {label} in {episode_path}")
    values = series.to_numpy(dtype=np.int64)
    if len(values) > 1 and np.any(np.diff(values) < 0):
        raise ValueError(f"{label} must be monotonically non-decreasing in {episode_path}")
    return values


def load_episode_arrays(episode: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(episode["path"])
    frame = pd.read_parquet(path / "timesteps.parquet", columns=list(REQUIRED_COLUMNS))
    if frame.empty:
        raise ValueError(f"Empty episode: {path}")
    timestep_indices = frame["timestep_index"].to_numpy(dtype=np.int64)
    if len(timestep_indices) > 1 and np.any(np.diff(timestep_indices) <= 0):
        raise ValueError(f"timestep_index must be strictly increasing in {path}")
    timestamps = frame["monotonic_timestamp_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps).all() or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0)):
        raise ValueError(f"monotonic_timestamp_s must be finite and strictly increasing in {path}")
    video_indices = _as_int_array(frame["overhead_video_frame_index"], "overhead_video_frame_index", path)
    camera_indices = _as_int_array(frame["overhead_camera_frame_index"], "overhead_camera_frame_index", path)

    left_names = [tuple(value) for value in frame["left_action_names"]]
    right_names = [tuple(value) for value in frame["right_action_names"]]
    if len(set(left_names)) != 1 or len(set(right_names)) != 1:
        raise ValueError(f"Action ordering changes within {path}")
    actions = np.stack(
        [
            np.concatenate((np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))
            for left, right in zip(
                frame["left_commanded_action"], frame["right_commanded_action"], strict=True
            )
        ]
    ).astype(np.float32)
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError(f"Actions must be a finite rank-2 array in {path}")

    # Orbit's intermediate and LeRobot exporters define one observation and
    # accepted command per recorder row. Camera matching is asynchronous, so
    # call order alone cannot justify shifting expert actions by one row.
    alignment_mode = "same_row_action"
    frame_ages = frame["overhead_frame_age_s"].to_numpy(dtype=np.float64)
    command_durations = np.asarray(
        [json.loads(raw).get("command_duration_s", np.nan) for raw in frame["metadata_json"]],
        dtype=np.float64,
    )
    timing_valid = np.isfinite(frame_ages) & np.isfinite(command_durations)

    reader = SequentialVideoReader(path / "videos/overhead.mp4")
    try:
        pixels = [reader.get(int(index)).copy() for index in video_indices]
    finally:
        reader.close()
    if any(image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 for image in pixels):
        raise ValueError(f"Overhead frames must be HWC uint8 RGB in {path}")

    per_step = {
        "pixels": pixels,
        "action": list(actions),
        "source_timestep_index": [np.asarray([v], dtype=np.float32) for v in timestep_indices],
        "source_camera_frame_index": [np.asarray([v], dtype=np.float32) for v in camera_indices],
        "source_video_frame_index": [np.asarray([v], dtype=np.float32) for v in video_indices],
    }
    stats = {
        "length": len(actions),
        "image_shape": list(pixels[0].shape),
        "action_dim": int(actions.shape[1]),
        "action_alignment": alignment_mode,
        "left_action_names": list(left_names[0]),
        "right_action_names": list(right_names[0]),
        "median_row_hz": float(1.0 / np.median(np.diff(timestamps))) if len(timestamps) > 1 else None,
        "duplicate_video_frame_fraction": float(1.0 - len(np.unique(video_indices)) / len(video_indices)),
        "camera_timing_count": int(timing_valid.sum()),
        "camera_frame_age_sum_s": float(frame_ages[timing_valid].sum()),
        "command_duration_sum_s": float(command_durations[timing_valid].sum()),
        "frame_age_exceeds_command_duration_count": int(
            np.count_nonzero(frame_ages[timing_valid] > command_durations[timing_valid])
        ),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_sum": actions.astype(np.float64).sum(axis=0).tolist(),
        "action_sum_sq": np.square(actions.astype(np.float64)).sum(axis=0).tolist(),
    }
    return per_step, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Orbit root, optionally COLLECTION_TYPE=/path (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--lewm-root", type=Path, default=Path("external/le-wm"))
    parser.add_argument("--stablewm-root", type=Path, default=Path("external/stable-worldmodel"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    specs = parse_dataset_specs(args.dataset)
    lewm_sha = git_sha(args.lewm_root)
    stablewm_sha = git_sha(args.stablewm_root)
    if lewm_sha is None or stablewm_sha is None:
        raise FileNotFoundError("Pinned LeWM and Stable World Model checkouts are required; run setup_lewm.sh")
    if lewm_sha != "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac":
        raise ValueError(f"Unexpected LeWM commit {lewm_sha}; rerun setup_lewm.sh or explicitly update the pin")
    if stablewm_sha != "9a66d7d020043c8efb507f45373e808714f0842d":
        raise ValueError(
            f"Unexpected Stable World Model commit {stablewm_sha}; rerun setup_lewm.sh or explicitly update the pin"
        )
    episodes = discover_episodes(specs)
    splits = stratified_episode_split(episodes, args.seed)
    split_by_uid = {uid: split for split, uids in splits.items() for uid in uids}
    output = args.output.expanduser().resolve()
    splits_path = (args.splits_output or output.with_suffix(".splits.json")).expanduser().resolve()
    manifest_path = (args.manifest_output or output.with_suffix(".manifest.json")).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    from stable_worldmodel.data import EPISODE_DATA_KEY, LanceWriter

    all_stats: list[dict[str, Any]] = []
    expected_schema: tuple[Any, ...] | None = None

    def converted() -> Iterator[dict[str, Any]]:
        nonlocal expected_schema
        for index, episode in enumerate(episodes, start=1):
            print(f"[{index}/{len(episodes)}] {episode['episode_uid']}", flush=True)
            per_step, stats = load_episode_arrays(episode)
            schema = (
                tuple(stats["image_shape"]),
                stats["action_dim"],
                tuple(stats["left_action_names"]),
                tuple(stats["right_action_names"]),
            )
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(f"Cross-episode RGB/action schema mismatch in {episode['episode_uid']}")
            all_stats.append({"episode_uid": episode["episode_uid"], **stats})
            per_step[EPISODE_DATA_KEY] = {
                "episode_uid": episode["episode_uid"],
                "source_dataset": episode["source_dataset"],
                "source_dataset_path": episode["source_dataset_path"],
                "source_episode_id": episode["source_episode_id"],
                "collection_type": episode["collection_type"],
                "split": split_by_uid[episode["episode_uid"]],
                "terminal_success_known": episode["terminal_success_known"],
                "terminal_success": episode["terminal_success"],
                "goal_eligible": episode["goal_eligible"],
                "action_alignment": stats["action_alignment"],
                "source_metadata_json": json.dumps(episode["metadata"], sort_keys=True),
            }
            yield per_step

    with LanceWriter(output, mode="overwrite", jpeg_quality=args.jpeg_quality) as writer:
        writer.write_episodes(converted())

    assert expected_schema is not None
    lengths = np.asarray([item["length"] for item in all_stats])
    action_dim = int(expected_schema[1])
    total = int(lengths.sum())
    action_sum = np.sum([item["action_sum"] for item in all_stats], axis=0)
    action_sum_sq = np.sum([item["action_sum_sq"] for item in all_stats], axis=0)
    action_mean = action_sum / total
    action_std = np.sqrt(np.maximum(action_sum_sq / total - action_mean**2, 0))
    timing_count = sum(item["camera_timing_count"] for item in all_stats)
    timing_diagnostic = {
        "rows_with_command_duration": timing_count,
        "mean_overhead_frame_age_s": (
            sum(item["camera_frame_age_sum_s"] for item in all_stats) / timing_count
            if timing_count
            else None
        ),
        "mean_command_duration_s": (
            sum(item["command_duration_sum_s"] for item in all_stats) / timing_count
            if timing_count
            else None
        ),
        "fraction_frame_age_exceeds_command_duration": (
            sum(item["frame_age_exceeds_command_duration_count"] for item in all_stats) / timing_count
            if timing_count
            else None
        ),
        "interpretation": (
            "Diagnostic only: asynchronous buffered camera matching and unrecorded post-command delay "
            "prevent exact exposure-versus-command ordering from being inferred from call order."
        ),
    }
    split_payload = {
        "version": 1,
        "seed": args.seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "splits": splits,
    }
    write_json(splits_path, split_payload)
    write_json(
        manifest_path,
        {
            "version": 1,
            "dataset": str(output),
            "sources": [{"collection_type": kind, "path": str(path)} for kind, path in specs],
            "orbit_git_sha": git_sha(Path(__file__).resolve().parents[1]),
            "lewm_git_sha": lewm_sha,
            "stable_worldmodel_git_sha": stablewm_sha,
            "split_manifest": str(splits_path),
            "num_episodes": len(episodes),
            "num_timesteps": total,
            "usable_raw_transitions": int(sum(max(0, value - 1) for value in lengths)),
            "trajectory_length": {
                "min": int(lengths.min()),
                "mean": float(lengths.mean()),
                "max": int(lengths.max()),
            },
            "rgb_shape": list(expected_schema[0]),
            "camera": "overhead",
            "action_dim": action_dim,
            "action_ordering": list(expected_schema[2]) + list(expected_schema[3]),
            "arm_ordering": "left then right",
            "gripper_dimensions": [len(expected_schema[2]) - 1, action_dim - 1],
            "action_semantics": "accepted absolute robot command after clipping/safety checks",
            "source_action_normalization": "none",
            "training_action_normalization": "zscore fit on train episodes only",
            "collection_type_counts": dict(Counter(ep["collection_type"] for ep in episodes)),
            "split_counts": {key: len(value) for key, value in splits.items()},
            "action_min": np.min([item["action_min"] for item in all_stats], axis=0).tolist(),
            "action_max": np.max([item["action_max"] for item in all_stats], axis=0).tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "median_episode_row_hz": float(np.median([s["median_row_hz"] for s in all_stats if s["median_row_hz"]])),
            "mean_duplicate_video_frame_fraction": float(
                np.mean([s["duplicate_video_frame_fraction"] for s in all_stats])
            ),
            "frameskip": 3,
            "effective_action_dim": 3 * action_dim,
            "causal_convention": "recorded observation[t] and same-row accepted action[t]",
            "camera_timing_diagnostic": timing_diagnostic,
            "hil_success_semantics": "HIL success=true is not interpreted as full-task success",
        },
    )
    print(f"Wrote {len(episodes)} episodes and {total} rows to {output}")


if __name__ == "__main__":
    main()
