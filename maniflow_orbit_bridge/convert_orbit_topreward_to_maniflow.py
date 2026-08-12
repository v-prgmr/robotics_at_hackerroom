"""Convert a multi-collection Orbit dataset and TOPReward results into one ManiFlow zarr."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

if __package__:
    from maniflow_orbit_bridge.convert_orbit_lerobot_to_maniflow import (
        TOPREWARD_SCORE_MODES,
        compute_topreward_action_arrays,
    )
    from maniflow_orbit_bridge.data_path_helpers import classify_collection_role, source_episode_progress
else:
    from convert_orbit_lerobot_to_maniflow import TOPREWARD_SCORE_MODES, compute_topreward_action_arrays
    from data_path_helpers import classify_collection_role, source_episode_progress


DEFAULT_CAMERAS = ("overhead", "left_wrist", "right_wrist")
PREFERRED_COLLECTION_ORDER = (
    "teabags_kitting_50_v2",
    "maniflow_hil_bounded",
    "successes",
    "failures",
)


@dataclass(frozen=True)
class EpisodeSpec:
    collection: str
    path: Path
    score_path: Path
    episode_id: str
    source_episode_index: int
    source_frames: int
    output_frames: int
    task: str
    source_role: str
    progress_valid: bool


class SequentialVideoFrameReader:
    """Read monotonically requested frames without caching complete videos."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.current_index = -1
        self.current_frame: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray:
        if index < 0:
            raise ValueError(f"Negative frame index {index} requested from {self.path}")
        if self.current_frame is not None and index == self.current_index:
            return self.current_frame
        if index < self.current_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.current_index = index - 1
            self.current_frame = None
        while self.current_index < index:
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {index} from video: {self.path}")
            self.current_index += 1
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        assert self.current_frame is not None
        return self.current_frame

    def close(self) -> None:
        self.capture.release()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _episode_number(episode_id: str) -> int:
    match = re.fullmatch(r"episode-(\d+)", episode_id)
    if match is None:
        raise ValueError(f"Unsupported Orbit episode id: {episode_id!r}")
    return int(match.group(1))


def _collection_paths(orbit_root: Path, selected: list[str] | None) -> list[Path]:
    available = {
        path.name: path
        for path in orbit_root.iterdir()
        if path.is_dir() and any(path.glob("episode-*/timesteps.parquet"))
    }
    if selected:
        missing = sorted(set(selected) - set(available))
        if missing:
            raise FileNotFoundError(f"Orbit collections not found: {missing}")
        names = selected
    else:
        names = [name for name in PREFERRED_COLLECTION_ORDER if name in available]
        names.extend(sorted(set(available) - set(names)))
    if not names:
        raise ValueError(f"No Orbit collections found under {orbit_root}")
    return [available[name] for name in names]


def discover_episodes(
    *,
    orbit_root: Path,
    topreward_results: Path,
    collections: list[str] | None,
    cameras: tuple[str, ...],
    frame_stride: int,
    max_episodes: int | None,
    score_mode: str,
    exponent_scale: float,
    max_weight: float,
) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    expected_dims: tuple[int, int] | None = None
    for collection_path in _collection_paths(orbit_root, collections):
        source_role, progress_valid = classify_collection_role(collection_path.name)
        episode_paths = sorted(path for path in collection_path.glob("episode-*") if path.is_dir())
        for episode_path in episode_paths:
            if max_episodes is not None and len(specs) >= max_episodes:
                return specs
            episode_id = episode_path.name
            metadata_path = episode_path / "episode_metadata.json"
            timestep_path = episode_path / "timesteps.parquet"
            if not metadata_path.exists() or not timestep_path.exists():
                raise FileNotFoundError(f"Incomplete Orbit episode: {episode_path}")

            metadata = _load_json(metadata_path)
            if metadata.get("episode_id") != episode_id:
                raise ValueError(f"Orbit episode provenance mismatch: {metadata_path}")
            timesteps = pd.read_parquet(timestep_path)
            if timesteps.empty:
                raise ValueError(f"Orbit episode has no timesteps: {episode_path}")
            if not np.array_equal(timesteps["timestep_index"].to_numpy(), np.arange(len(timesteps))):
                raise ValueError(f"Orbit timestep indices must be contiguous from zero: {episode_path}")

            state_dim = len(timesteps.iloc[0]["left_follower_joints"]) + len(
                timesteps.iloc[0]["right_follower_joints"]
            )
            action_dim = len(timesteps.iloc[0]["left_commanded_action"]) + len(
                timesteps.iloc[0]["right_commanded_action"]
            )
            dims = (state_dim, action_dim)
            if expected_dims is None:
                expected_dims = dims
            elif dims != expected_dims:
                raise ValueError(f"State/action dimensions differ across collections: {dims} != {expected_dims}")

            for camera in cameras:
                frame_column = f"{camera}_video_frame_index"
                video_path = episode_path / "videos" / f"{camera}.mp4"
                if frame_column not in timesteps.columns or bool(timesteps[frame_column].isna().any()):
                    raise ValueError(f"Missing camera frame indices for {camera}: {episode_path}")
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing camera video: {video_path}")

            score_path = topreward_results / "episodes" / collection_path.name / f"{episode_id}.json"
            if not score_path.exists():
                raise FileNotFoundError(f"Missing TOPReward result: {score_path}")
            score = _load_json(score_path)
            if score.get("status") != "complete":
                raise ValueError(f"TOPReward result is not complete: {score_path}")
            if score.get("dataset") != collection_path.name or score.get("episode_id") != episode_id:
                raise ValueError(f"TOPReward provenance mismatch: {score_path}")
            if int(score.get("num_timesteps", -1)) != len(timesteps):
                raise ValueError(f"TOPReward frame count mismatch: {score_path}")
            compute_topreward_action_arrays(
                total_frames=len(timesteps),
                anchors=score["anchors"],
                score_mode=score_mode,
                exponent_scale=exponent_scale,
                max_weight=max_weight,
            )

            specs.append(
                EpisodeSpec(
                    collection=collection_path.name,
                    path=episode_path,
                    score_path=score_path,
                    episode_id=episode_id,
                    source_episode_index=_episode_number(episode_id),
                    source_frames=len(timesteps),
                    output_frames=(len(timesteps) + frame_stride - 1) // frame_stride,
                    task=str(metadata.get("task_description", "")),
                    source_role=source_role,
                    progress_valid=progress_valid,
                )
            )
    return specs


def _stack_vectors(timesteps: pd.DataFrame, left: str, right: str) -> np.ndarray:
    return np.stack(
        [
            np.concatenate((np.asarray(left_value), np.asarray(right_value)))
            for left_value, right_value in zip(timesteps[left], timesteps[right], strict=True)
        ]
    ).astype(np.float32, copy=False)


def _batch_ranges(total: int, global_start: int, batch_size: int, chunk_length: int):
    """Yield local ranges while keeping interior writes aligned to zarr chunks."""
    position = 0
    while position < total:
        absolute = global_start + position
        offset = absolute % chunk_length
        if offset:
            count = min(total - position, chunk_length - offset)
        else:
            aligned_batch = max(chunk_length, (batch_size // chunk_length) * chunk_length)
            count = min(total - position, aligned_batch)
        yield position, position + count
        position += count


def _write_episode_camera(
    *,
    output_zarr: str,
    episode_path: str,
    camera: str,
    frame_indices: np.ndarray,
    write_start: int,
    image_size: int,
    batch_size: int,
    chunk_length: int,
) -> dict[str, Any]:
    """Decode one camera sequentially and write lossless image batches."""
    import zarr

    cv2.setNumThreads(1)
    episode_dir = Path(episode_path)
    reader = SequentialVideoFrameReader(episode_dir / "videos" / f"{camera}.mp4")
    target = zarr.open_group(output_zarr, mode="r+")[f"data/{camera}"]
    started = time.perf_counter()
    try:
        for local_start, local_end in _batch_ranges(
            len(frame_indices), write_start, batch_size, chunk_length
        ):
            batch = np.empty((local_end - local_start, 3, image_size, image_size), dtype=np.uint8)
            for output_offset, frame_index in enumerate(frame_indices[local_start:local_end]):
                frame = reader.get(int(frame_index))
                resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                batch[output_offset] = np.moveaxis(resized, -1, 0)
            target[write_start + local_start : write_start + local_end] = batch
    finally:
        reader.close()
    return {
        "camera": camera,
        "frames": len(frame_indices),
        "seconds": time.perf_counter() - started,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def _completed_episode_indexes(
    state: dict[str, Any], conversion_config: dict[str, Any], episode_count: int
) -> set[int]:
    if state.get("config") != conversion_config:
        raise ValueError("Resume configuration does not match the existing conversion state")
    completed = {int(index) for index in state.get("completed_episodes", [])}
    if any(index < 0 or index >= episode_count for index in completed):
        raise ValueError("Resume state contains an invalid completed episode index")
    return completed


def convert(
    *,
    orbit_root: Path,
    topreward_results: Path,
    output_zarr: Path,
    collections: list[str] | None,
    cameras: tuple[str, ...],
    image_size: int,
    frame_stride: int,
    max_episodes: int | None,
    overwrite: bool,
    chunk_length: int,
    score_mode: str,
    exponent_scale: float | None,
    max_weight: float,
    video_workers: int = 3,
    write_batch_size: int = 256,
    compression_level: int = 1,
    resume: bool = False,
) -> None:
    import numcodecs
    import zarr

    orbit_root = orbit_root.expanduser().resolve()
    topreward_results = topreward_results.expanduser().resolve()
    output_zarr = output_zarr.expanduser().resolve()
    if not orbit_root.is_dir():
        raise NotADirectoryError(f"Orbit root does not exist: {orbit_root}")
    if not topreward_results.is_dir():
        raise NotADirectoryError(f"TOPReward results do not exist: {topreward_results}")
    if image_size <= 0 or frame_stride <= 0 or chunk_length <= 0:
        raise ValueError("image_size, frame_stride, and chunk_length must be positive")
    if video_workers <= 0 or write_batch_size <= 0:
        raise ValueError("video_workers and write_batch_size must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if score_mode not in TOPREWARD_SCORE_MODES:
        raise ValueError(f"Unsupported TOPReward score mode: {score_mode!r}")
    if exponent_scale is None:
        exponent_scale = 0.2 if score_mode == "raw_logp_true" else 2.0

    specs = discover_episodes(
        orbit_root=orbit_root,
        topreward_results=topreward_results,
        collections=collections,
        cameras=cameras,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        score_mode=score_mode,
        exponent_scale=exponent_scale,
        max_weight=max_weight,
    )
    if not specs:
        raise ValueError("No episodes selected for conversion")

    first_timesteps = pd.read_parquet(specs[0].path / "timesteps.parquet")
    state_dim = len(first_timesteps.iloc[0]["left_follower_joints"]) + len(
        first_timesteps.iloc[0]["right_follower_joints"]
    )
    action_dim = len(first_timesteps.iloc[0]["left_commanded_action"]) + len(
        first_timesteps.iloc[0]["right_commanded_action"]
    )
    episode_ends = np.cumsum(np.asarray([spec.output_frames for spec in specs], dtype=np.int64))
    total_frames = int(episode_ends[-1])
    collection_names = list(dict.fromkeys(spec.collection for spec in specs))
    collection_indices = {name: index for index, name in enumerate(collection_names)}
    task_names = list(dict.fromkeys(spec.task for spec in specs))
    task_indices = {task: index for index, task in enumerate(task_names)}
    episode_provenance = {
        str(output_episode_index): {
            "collection": spec.collection,
            "episode_id": spec.episode_id,
            "source_frames": spec.source_frames,
            "output_frames": spec.output_frames,
            "source_role": spec.source_role,
            "progress_valid": spec.progress_valid,
        }
        for output_episode_index, spec in enumerate(specs)
    }
    conversion_config = {
        "version": 1,
        "source_orbit_root": str(orbit_root),
        "source_topreward_results": str(topreward_results),
        "collections": collection_names,
        "cameras": list(cameras),
        "image_size": image_size,
        "frame_stride": frame_stride,
        "chunk_length": chunk_length,
        "write_batch_size": write_batch_size,
        "compression": {"name": "zstd", "level": compression_level, "shuffle": "bitshuffle"},
        "state_dim": state_dim,
        "action_dim": action_dim,
        "score_mode": score_mode,
        "exponent_scale": exponent_scale,
        "max_weight": max_weight,
        "episodes": episode_provenance,
    }
    state_path = output_zarr / ".conversion_state.json"

    if resume:
        if not output_zarr.is_dir() or not state_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without {state_path}. The output must have been created by the optimized converter."
            )
        state = _load_json(state_path)
        completed_episodes = _completed_episode_indexes(state, conversion_config, len(specs))
        root = zarr.open_group(str(output_zarr), mode="r+")
        if not np.array_equal(root["meta/episode_ends"][:], episode_ends):
            raise ValueError("Resume episode boundaries do not match the existing zarr")
        print(f"Resuming with {len(completed_episodes)}/{len(specs)} episodes complete", flush=True)
    else:
        if output_zarr.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_zarr}. Pass --overwrite to replace it or --resume to continue it."
                )
            shutil.rmtree(output_zarr)
        output_zarr.parent.mkdir(parents=True, exist_ok=True)

        compressor = numcodecs.Blosc(
            cname="zstd", clevel=compression_level, shuffle=numcodecs.Blosc.BITSHUFFLE
        )
        root = zarr.open_group(str(output_zarr), mode="w")
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")
        vector_chunks = (min(chunk_length, total_frames),)
        for camera in cameras:
            data_group.zeros(
                camera,
                shape=(total_frames, 3, image_size, image_size),
                chunks=(min(chunk_length, total_frames), 3, image_size, image_size),
                dtype=np.uint8,
                compressor=compressor,
            )
        data_group.zeros(
            "state",
            shape=(total_frames, state_dim),
            chunks=(min(chunk_length, total_frames), state_dim),
            dtype=np.float32,
            compressor=compressor,
        )
        data_group.zeros(
            "action",
            shape=(total_frames, action_dim),
            chunks=(min(chunk_length, total_frames), action_dim),
            dtype=np.float32,
            compressor=compressor,
        )
        for name, dtype, fill_value in (
            ("task_index", np.int64, 0),
            ("delta_score", np.float32, 0),
            ("weight_unclipped", np.float32, 1),
            ("topreward_weight", np.float32, 1),
            ("action_valid", np.bool_, 1),
            ("source_collection_index", np.int64, 0),
            ("source_episode_index", np.int64, 0),
            ("source_frame_index", np.int64, 0),
            ("episode_progress", np.float32, 0),
            ("progress_valid", np.bool_, 0),
        ):
            data_group.full(
                name,
                fill_value=fill_value,
                shape=(total_frames,),
                chunks=vector_chunks,
                dtype=dtype,
                compressor=compressor,
            )
        meta_group.array("episode_ends", data=episode_ends, chunks=episode_ends.shape, dtype=np.int64)
        completed_episodes: set[int] = set()
        state = {"status": "in_progress", "config": conversion_config, "completed_episodes": []}
        _atomic_write_json(state_path, state)

    array_names = (
        "state",
        "action",
        "task_index",
        "delta_score",
        "weight_unclipped",
        "topreward_weight",
        "action_valid",
        "source_collection_index",
        "source_episode_index",
        "source_frame_index",
        "episode_progress",
        "progress_valid",
    )
    arrays = {name: root[f"data/{name}"] for name in array_names}
    conversion_started = time.perf_counter()
    max_workers = min(video_workers, len(cameras))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for output_episode_index, spec in enumerate(specs):
            if output_episode_index in completed_episodes:
                continue
            episode_started = time.perf_counter()
            write_start = 0 if output_episode_index == 0 else int(episode_ends[output_episode_index - 1])
            write_end = int(episode_ends[output_episode_index])
            timesteps = pd.read_parquet(spec.path / "timesteps.parquet").iloc[::frame_stride].reset_index(drop=True)
            score = _load_json(spec.score_path)
            delta_score, weight_unclipped, weight = compute_topreward_action_arrays(
                total_frames=spec.source_frames,
                anchors=score["anchors"],
                score_mode=score_mode,
                exponent_scale=exponent_scale,
                max_weight=max_weight,
            )
            kept = np.arange(0, spec.source_frames, frame_stride)
            source_progress, source_valid = source_episode_progress(
                spec.source_frames, valid=spec.progress_valid
            )
            camera_futures = [
                executor.submit(
                    _write_episode_camera,
                    output_zarr=str(output_zarr),
                    episode_path=str(spec.path),
                    camera=camera,
                    frame_indices=timesteps[f"{camera}_video_frame_index"].to_numpy(dtype=np.int64),
                    write_start=write_start,
                    image_size=image_size,
                    batch_size=write_batch_size,
                    chunk_length=chunk_length,
                )
                for camera in cameras
            ]

            arrays["state"][write_start:write_end] = _stack_vectors(
                timesteps, "left_follower_joints", "right_follower_joints"
            )
            arrays["action"][write_start:write_end] = _stack_vectors(
                timesteps, "left_commanded_action", "right_commanded_action"
            )
            arrays["task_index"][write_start:write_end] = task_indices[spec.task]
            arrays["delta_score"][write_start:write_end] = delta_score[kept]
            arrays["weight_unclipped"][write_start:write_end] = weight_unclipped[kept]
            arrays["topreward_weight"][write_start:write_end] = weight[kept]
            arrays["source_collection_index"][write_start:write_end] = collection_indices[spec.collection]
            arrays["source_episode_index"][write_start:write_end] = spec.source_episode_index
            arrays["source_frame_index"][write_start:write_end] = timesteps["timestep_index"].to_numpy(
                dtype=np.int64
            )
            arrays["episode_progress"][write_start:write_end] = source_progress[kept]
            arrays["progress_valid"][write_start:write_end] = source_valid[kept]
            camera_results = [future.result() for future in camera_futures]

            completed_episodes.add(output_episode_index)
            state["completed_episodes"] = sorted(completed_episodes)
            _atomic_write_json(state_path, state)
            camera_seconds = max(float(result["seconds"]) for result in camera_results)
            print(
                f"episode {output_episode_index + 1}/{len(specs)}: {spec.collection}/{spec.episode_id} "
                f"frames={spec.output_frames} elapsed={time.perf_counter() - episode_started:.1f}s "
                f"slowest_camera={camera_seconds:.1f}s",
                flush=True,
            )

    sidecar = {
        "task_names": {str(index): task for index, task in enumerate(task_names)},
        "camera_features": {camera: f"{camera}_video_frame_index" for camera in cameras},
        "source_orbit_root": str(orbit_root),
        "source_topreward_results": str(topreward_results),
        "collection_names": {str(index): name for index, name in enumerate(collection_names)},
        "episodes": episode_provenance,
        "collection_roles": {
            name: {
                "source_role": classify_collection_role(name)[0],
                "progress_valid": classify_collection_role(name)[1],
            }
            for name in collection_names
        },
        "progress": {
            "target": "full-episode normalized temporal progress",
            "formula": "source_frame_index / (source_episode_frames - 1); singleton = 0",
            "computed_before_frame_stride": True,
            "phase_1_supervised_roles": ["expert", "success"],
            "phase_1_excluded_roles": ["hil", "correction", "failure"],
        },
        "image_size": image_size,
        "frame_stride": frame_stride,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "conversion": {
            "video_workers": max_workers,
            "write_batch_size": write_batch_size,
            "chunk_length": chunk_length,
            "compression": conversion_config["compression"],
        },
        "topreward": {
            "enabled": True,
            "score_source": "logp_true",
            "score_mode": score_mode,
            "exponent_scale": exponent_scale,
            "max_weight": max_weight,
            "minimum_weight": None,
            "interval": "action[a_prev:a_cur]",
            "outside_interval": {
                "delta_score": 0.0,
                "weight_unclipped": 1.0,
                "topreward_weight": 1.0,
            },
        },
    }
    _atomic_write_json(output_zarr / "orbit_tasks.json", sidecar)
    state["status"] = "complete"
    state["elapsed_seconds"] = time.perf_counter() - conversion_started
    _atomic_write_json(state_path, state)
    print(
        f"Converted {len(specs)} episodes and {total_frames} frames into {output_zarr} "
        f"in {state['elapsed_seconds']:.1f}s",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orbit-root", type=Path, required=True, help="Container with Orbit collection directories.")
    parser.add_argument("--topreward-results", type=Path, required=True, help="TOPReward results containing episodes/.")
    parser.add_argument("--output-zarr", type=Path, required=True, help="Destination merged ManiFlow zarr.")
    parser.add_argument("--collection", action="append", help="Collection to include. Defaults to all collections.")
    parser.add_argument("--camera", action="append", help="Camera name. Defaults to all three Orbit cameras.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, help="Global episode limit for smoke tests.")
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--video-workers", type=int, default=3, help="Parallel camera decoder processes.")
    parser.add_argument("--write-batch-size", type=int, default=256, help="Decoded images buffered per zarr write.")
    parser.add_argument("--compression-level", type=int, default=1, help="Lossless zstd level from 0 to 9.")
    parser.add_argument("--topreward-score-mode", choices=TOPREWARD_SCORE_MODES, default="raw_logp_true")
    parser.add_argument("--topreward-exponent-scale", type=float)
    parser.add_argument("--topreward-max-weight", type=float, default=2.0)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    output_mode.add_argument("--resume", action="store_true", help="Continue an optimized interrupted conversion.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert(
        orbit_root=args.orbit_root,
        topreward_results=args.topreward_results,
        output_zarr=args.output_zarr,
        collections=args.collection,
        cameras=tuple(args.camera or DEFAULT_CAMERAS),
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        chunk_length=args.chunk_length,
        score_mode=args.topreward_score_mode,
        exponent_scale=args.topreward_exponent_scale,
        max_weight=args.topreward_max_weight,
        video_workers=args.video_workers,
        write_batch_size=args.write_batch_size,
        compression_level=args.compression_level,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
