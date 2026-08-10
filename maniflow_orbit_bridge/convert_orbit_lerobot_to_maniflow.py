"""Convert an Orbit LeRobot v3 export into a ManiFlow replay-buffer zarr."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_CAMERAS = {
    "overhead": "observation.images.overhead",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}

TOPREWARD_SCORE_MODES = ("raw_logp_true", "normalized_progress")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_parquet_tree(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _parse_camera_arg(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERAS)

    cameras: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("Camera entries must be NAME=LEROBOT_FEATURE")
        name, feature = value.split("=", 1)
        name = name.strip()
        feature = feature.strip()
        if not name or not feature:
            raise argparse.ArgumentTypeError("Camera entries must be NAME=LEROBOT_FEATURE")
        cameras[name] = feature
    return cameras


def _episode_task(row: pd.Series) -> str:
    tasks = row.get("tasks", [])
    if isinstance(tasks, str):
        return tasks
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)) and tasks:
        return str(tasks[0])
    return ""


def _stack_vectors(series: pd.Series, *, key: str) -> np.ndarray:
    try:
        return np.stack(series.to_numpy()).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(f"Could not stack vector column {key!r}") from exc


def _video_path(root: Path, info: dict[str, Any], video_key: str, episode: pd.Series) -> Path:
    chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
    file_index = int(episode[f"videos/{video_key}/file_index"])
    template = info.get("video_path")
    if not template:
        raise ValueError("LeRobot info.json does not contain video_path")
    return root / template.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)


def _write_video_frames(
    *,
    source_path: Path,
    target_array: Any,
    target_start: int,
    source_start_frame: int,
    source_frame_count: int,
    frame_stride: int,
    image_size: int,
) -> int:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        written = 0
        for frame_offset in range(source_frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                absolute = source_start_frame + frame_offset
                raise ValueError(f"Could not read frame {absolute} from {source_path}")
            if frame_offset % frame_stride != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            target_array[target_start + written] = np.moveaxis(frame_rgb, -1, 0)
            written += 1
    finally:
        capture.release()

    return written


def _episode_output_length(length: int, frame_stride: int) -> int:
    return (length + frame_stride - 1) // frame_stride


def compute_topreward_action_arrays(
    *,
    total_frames: int,
    anchors: list[dict[str, Any]],
    score_mode: str,
    exponent_scale: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast consecutive observation-anchor deltas onto causal actions."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if score_mode not in TOPREWARD_SCORE_MODES:
        raise ValueError(f"Unsupported TOPReward score mode: {score_mode!r}")
    if not math.isfinite(exponent_scale) or exponent_scale < 0:
        raise ValueError("TOPReward exponent scale must be non-negative")
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("TOPReward max weight must be positive")
    if len(anchors) < 2:
        raise ValueError("TOPReward requires at least two anchors per episode")

    ordered = sorted(anchors, key=lambda anchor: int(anchor["anchor_index"]))
    anchor_indices = np.asarray([int(anchor["anchor_timestep_index"]) for anchor in ordered], dtype=np.int64)
    raw_scores = np.asarray([float(anchor["logp_true"]) for anchor in ordered], dtype=np.float64)
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("TOPReward anchor logp_true values must be finite")
    if anchor_indices[0] != 0 or anchor_indices[-1] != total_frames - 1:
        raise ValueError(
            "TOPReward anchors must cover the episode observation endpoints: "
            f"expected [0, {total_frames - 1}], got [{anchor_indices[0]}, {anchor_indices[-1]}]"
        )
    if np.any(np.diff(anchor_indices) <= 0):
        raise ValueError("TOPReward anchor timestep indices must be strictly increasing")

    if score_mode == "normalized_progress":
        score_range = float(raw_scores.max() - raw_scores.min())
        scores = (raw_scores - raw_scores.min()) / score_range if score_range > 0 else np.zeros_like(raw_scores)
    else:
        scores = raw_scores

    delta_score = np.zeros(total_frames, dtype=np.float32)
    weight_unclipped = np.ones(total_frames, dtype=np.float32)
    weight = np.ones(total_frames, dtype=np.float32)
    for previous, current, delta in zip(anchor_indices[:-1], anchor_indices[1:], np.diff(scores), strict=True):
        unclipped = math.exp(exponent_scale * float(delta))
        delta_score[previous:current] = delta
        weight_unclipped[previous:current] = unclipped
        weight[previous:current] = min(unclipped, max_weight)

    return delta_score, weight_unclipped, weight


def _load_topreward_episode(
    results_root: Path,
    dataset_name: str,
    episode_index: int,
    expected_frames: int,
) -> dict[str, Any]:
    episode_id = f"episode-{episode_index + 1:06d}"
    path = results_root / "episodes" / dataset_name / f"{episode_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing TOPReward episode result: {path}")
    payload = _load_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"TOPReward result is not complete: {path}")
    if payload.get("dataset") != dataset_name or payload.get("episode_id") != episode_id:
        raise ValueError(f"TOPReward episode provenance mismatch: {path}")
    if int(payload.get("num_timesteps", -1)) != expected_frames:
        raise ValueError(
            f"TOPReward frame count mismatch for {episode_id}: "
            f"expected {expected_frames}, got {payload.get('num_timesteps')}"
        )
    return payload


def convert(
    *,
    lerobot_root: Path,
    output_zarr: Path,
    cameras: dict[str, str],
    image_size: int,
    frame_stride: int,
    max_episodes: int | None,
    overwrite: bool,
    chunk_length: int,
    topreward_results: Path | None = None,
    topreward_dataset: str | None = None,
    topreward_score_mode: str = "raw_logp_true",
    topreward_exponent_scale: float | None = None,
    topreward_max_weight: float = 2.0,
) -> None:
    import numcodecs
    import zarr

    lerobot_root = lerobot_root.expanduser().resolve()
    output_zarr = output_zarr.expanduser().resolve()
    if topreward_results is not None:
        topreward_results = topreward_results.expanduser().resolve()
        if not topreward_results.is_dir():
            raise NotADirectoryError(f"TOPReward results directory does not exist: {topreward_results}")
        if not topreward_dataset:
            raise ValueError("--topreward-dataset is required with --topreward-results")
    if topreward_score_mode not in TOPREWARD_SCORE_MODES:
        raise ValueError(f"Unsupported TOPReward score mode: {topreward_score_mode!r}")
    if topreward_exponent_scale is None:
        topreward_exponent_scale = 2.0 if topreward_score_mode == "normalized_progress" else 0.2

    info_path = lerobot_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Expected LeRobot metadata at {info_path}")

    info = _load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0 export, got {info.get('codebase_version')!r}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    features = info.get("features", {})
    for target_name, source_feature in cameras.items():
        feature = features.get(source_feature)
        if feature is None:
            raise ValueError(f"Camera feature {source_feature!r} for {target_name!r} is missing from info.json")
        if feature.get("dtype") != "video":
            raise ValueError(f"Camera feature {source_feature!r} is not a video feature")

    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_zarr}. Pass --overwrite to replace it.")
        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    data = _read_parquet_tree(lerobot_root / "data")
    episodes = _read_parquet_tree(lerobot_root / "meta/episodes")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes.iloc[:max_episodes].reset_index(drop=True)
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    output_lengths: list[int] = []
    episode_task_by_index: dict[int, str] = {}
    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        output_lengths.append(_episode_output_length(length, frame_stride))
        episode_task_by_index[episode_index] = _episode_task(episode)

    episode_ends = np.cumsum(np.asarray(output_lengths, dtype=np.int64))
    total_frames = int(episode_ends[-1]) if len(episode_ends) else 0
    if total_frames == 0:
        raise ValueError("No frames to convert")

    state_dim = len(data["observation.state"].iloc[0])
    action_dim = len(data["action"].iloc[0])
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    root = zarr.open_group(str(output_zarr), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    image_chunks = (min(chunk_length, total_frames), 3, image_size, image_size)
    vector_chunks = (min(chunk_length, total_frames),)
    camera_arrays = {
        name: data_group.zeros(
            name,
            shape=(total_frames, 3, image_size, image_size),
            chunks=image_chunks,
            dtype=np.uint8,
            compressor=compressor,
        )
        for name in cameras
    }
    state_array = data_group.zeros(
        "state",
        shape=(total_frames, state_dim),
        chunks=(min(chunk_length, total_frames), state_dim),
        dtype=np.float32,
        compressor=compressor,
    )
    action_array = data_group.zeros(
        "action",
        shape=(total_frames, action_dim),
        chunks=(min(chunk_length, total_frames), action_dim),
        dtype=np.float32,
        compressor=compressor,
    )
    task_index_array = data_group.zeros(
        "task_index",
        shape=(total_frames,),
        chunks=vector_chunks,
        dtype=np.int64,
        compressor=compressor,
    )
    delta_score_array = data_group.zeros(
        "delta_score", shape=(total_frames,), chunks=vector_chunks, dtype=np.float32, compressor=compressor
    )
    weight_unclipped_array = data_group.ones(
        "weight_unclipped", shape=(total_frames,), chunks=vector_chunks, dtype=np.float32, compressor=compressor
    )
    topreward_weight_array = data_group.ones(
        "topreward_weight", shape=(total_frames,), chunks=vector_chunks, dtype=np.float32, compressor=compressor
    )
    action_valid_array = data_group.ones(
        "action_valid", shape=(total_frames,), chunks=vector_chunks, dtype=np.bool_, compressor=compressor
    )
    source_episode_index_array = data_group.zeros(
        "source_episode_index", shape=(total_frames,), chunks=vector_chunks, dtype=np.int64, compressor=compressor
    )
    source_frame_index_array = data_group.zeros(
        "source_frame_index", shape=(total_frames,), chunks=vector_chunks, dtype=np.int64, compressor=compressor
    )
    meta_group.array("episode_ends", data=episode_ends, chunks=episode_ends.shape, dtype=np.int64)

    task_names: dict[int, str] = {}
    source_episode_ids: dict[int, str] = {}
    write_start = 0
    fps = int(info["fps"])
    for episode_offset, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        source_length = int(episode["length"])
        output_length = output_lengths[episode_offset]
        write_end = write_start + output_length

        episode_data = data[data["episode_index"] == episode_index].copy()
        if len(episode_data) != source_length:
            raise ValueError(
                f"Episode {episode_index} length mismatch: metadata={source_length}, data={len(episode_data)}"
            )
        episode_data = episode_data.sort_values("frame_index").iloc[::frame_stride]
        if len(episode_data) != output_length:
            raise ValueError(f"Episode {episode_index} output length mismatch after stride")

        state_array[write_start:write_end] = _stack_vectors(episode_data["observation.state"], key="observation.state")
        action_array[write_start:write_end] = _stack_vectors(episode_data["action"], key="action")
        source_episode_index_array[write_start:write_end] = episode_index
        source_frame_index_array[write_start:write_end] = episode_data["frame_index"].to_numpy(dtype=np.int64)

        if topreward_results is not None:
            payload = _load_topreward_episode(
                topreward_results,
                str(topreward_dataset),
                episode_index,
                source_length,
            )
            delta_score, weight_unclipped, topreward_weight = compute_topreward_action_arrays(
                total_frames=source_length,
                anchors=payload["anchors"],
                score_mode=topreward_score_mode,
                exponent_scale=topreward_exponent_scale,
                max_weight=topreward_max_weight,
            )
            kept = np.arange(0, source_length, frame_stride)
            delta_score_array[write_start:write_end] = delta_score[kept]
            weight_unclipped_array[write_start:write_end] = weight_unclipped[kept]
            topreward_weight_array[write_start:write_end] = topreward_weight[kept]
            source_episode_ids[episode_index] = str(payload["episode_id"])

        task_index = int(episode_data["task_index"].iloc[0])
        task_index_array[write_start:write_end] = task_index
        task_names[task_index] = episode_task_by_index.get(episode_index, "")

        for target_name, source_feature in cameras.items():
            source_path = _video_path(lerobot_root, info, source_feature, episode)
            source_start_frame = int(round(float(episode[f"videos/{source_feature}/from_timestamp"]) * fps))
            written = _write_video_frames(
                source_path=source_path,
                target_array=camera_arrays[target_name],
                target_start=write_start,
                source_start_frame=source_start_frame,
                source_frame_count=source_length,
                frame_stride=frame_stride,
                image_size=image_size,
            )
            if written != output_length:
                raise ValueError(
                    f"Episode {episode_index} camera {target_name!r} wrote {written} frames, expected {output_length}"
                )

        print(
            f"episode {episode_index}: source_frames={source_length} output_frames={output_length} "
            f"task_index={task_index}"
        )
        write_start = write_end

    task_sidecar = {
        "task_names": {str(index): task for index, task in sorted(task_names.items())},
        "camera_features": cameras,
        "source_lerobot_root": str(lerobot_root),
        "image_size": image_size,
        "frame_stride": frame_stride,
        "fps": fps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "source_episode_ids": {str(index): episode_id for index, episode_id in sorted(source_episode_ids.items())},
        "topreward": {
            "enabled": topreward_results is not None,
            "results_root": str(topreward_results) if topreward_results is not None else None,
            "dataset": topreward_dataset,
            "score_source": "logp_true",
            "score_mode": topreward_score_mode,
            "exponent_scale": topreward_exponent_scale,
            "max_weight": topreward_max_weight,
            "minimum_weight": None,
            "interval": "action[a_prev:a_cur]",
            "outside_interval": {
                "delta_score": 0.0,
                "weight_unclipped": 1.0,
                "topreward_weight": 1.0,
            },
        },
    }
    with (output_zarr / "orbit_tasks.json").open("w", encoding="utf-8") as file:
        json.dump(task_sidecar, file, indent=2, ensure_ascii=False)

    print(f"Converted {len(episodes)} episodes, {total_frames} frames")
    print(f"Output: {output_zarr}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True, help="Orbit LeRobot v3 export root.")
    parser.add_argument("--output-zarr", type=Path, required=True, help="Destination ManiFlow .zarr directory.")
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera mapping NAME=LEROBOT_FEATURE. Defaults to overhead/left_wrist/right_wrist.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square RGB image size written to zarr.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame from each episode.")
    parser.add_argument("--max-episodes", type=int, help="Convert only the first N episodes for smoke tests.")
    parser.add_argument("--chunk-length", type=int, default=64, help="Zarr chunk length along time dimension.")
    parser.add_argument("--topreward-results", type=Path, help="TOPReward output directory containing episodes/.")
    parser.add_argument("--topreward-dataset", help="Dataset key under TOPReward results/episodes/.")
    parser.add_argument(
        "--topreward-score-mode",
        choices=TOPREWARD_SCORE_MODES,
        default="raw_logp_true",
        help="Use raw logp_true deltas (paper baseline) or episode-normalized progress (diagnostic ablation).",
    )
    parser.add_argument(
        "--topreward-exponent-scale",
        type=float,
        help="Exponent scale. Defaults to 0.2 for raw logp_true and 2.0 for normalized progress.",
    )
    parser.add_argument("--topreward-max-weight", type=float, default=2.0, help="Upper weight cap.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output zarr if it exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert(
        lerobot_root=args.lerobot_root,
        output_zarr=args.output_zarr,
        cameras=_parse_camera_arg(args.camera),
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        chunk_length=args.chunk_length,
        topreward_results=args.topreward_results,
        topreward_dataset=args.topreward_dataset,
        topreward_score_mode=args.topreward_score_mode,
        topreward_exponent_scale=args.topreward_exponent_scale,
        topreward_max_weight=args.topreward_max_weight,
    )


if __name__ == "__main__":
    main()
