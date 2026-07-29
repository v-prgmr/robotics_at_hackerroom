"""Convert an Orbit LeRobot v3 export into the LeRobot v2.1 layout used by OpenPI.

OpenPI's current training loader depends on a LeRobot version that expects:

    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/<video_key>/episode_000000.mp4

Orbit's standard exporter writes the newer chunked v3 layout. This script splits the v3 data/video files into
the older per-episode layout without changing state/action values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_CHUNK_SIZE = 1000
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _episode_chunk(episode_index: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> int:
    return episode_index // chunks_size


def _read_parquet_tree(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _tasks_from_episodes(episodes: pd.DataFrame) -> list[str]:
    tasks: list[str] = []
    for value in episodes["tasks"]:
        episode_tasks = list(value) if not isinstance(value, str) else [value]
        for task in episode_tasks:
            if task not in tasks:
                tasks.append(task)
    return tasks


def _scalar_feature_stats(data: pd.DataFrame, scalar_features: list[str]) -> dict[str, dict[str, list[Any]]]:
    stats: dict[str, dict[str, list[Any]]] = {}
    for key in scalar_features:
        values = np.asarray(data[key].tolist() if data[key].dtype == object else data[key].to_numpy())
        if values.ndim == 1:
            keepdims = True
        else:
            keepdims = False
        stats[key] = {
            "min": np.min(values, axis=0, keepdims=keepdims).tolist(),
            "max": np.max(values, axis=0, keepdims=keepdims).tolist(),
            "mean": np.mean(values, axis=0, keepdims=keepdims).tolist(),
            "std": np.std(values, axis=0, keepdims=keepdims).tolist(),
            "count": [int(len(values))],
        }
    return stats


def _copy_video_segment(
    source_path: Path,
    output_path: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open source video: {source_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {start_frame} from {source_path}")

        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open output video writer: {output_path}")

        try:
            writer.write(first_frame)
            for frame_offset in range(1, frame_count):
                ok, frame = capture.read()
                if not ok:
                    absolute_frame = start_frame + frame_offset
                    raise ValueError(f"Could not read frame {absolute_frame} from {source_path}")
                writer.write(frame)
        finally:
            writer.release()
    finally:
        capture.release()


def _write_episode_videos(source_root: Path, output_root: Path, episodes: pd.DataFrame, video_keys: list[str], fps: int) -> None:
    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        episode_chunk = _episode_chunk(episode_index)
        frame_count = int(episode["length"])
        for video_key in video_keys:
            source_chunk = int(episode[f"videos/{video_key}/chunk_index"])
            source_file = int(episode[f"videos/{video_key}/file_index"])
            from_timestamp = float(episode[f"videos/{video_key}/from_timestamp"])
            source_path = source_root / f"videos/{video_key}/chunk-{source_chunk:03d}/file-{source_file:03d}.mp4"
            output_path = output_root / V21_VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            start_frame = int(round(from_timestamp * fps))
            print(f"Writing {output_path.relative_to(output_root)}")
            _copy_video_segment(
                source_path,
                output_path,
                start_frame=start_frame,
                frame_count=frame_count,
                fps=fps,
            )


def export_openpi_lerobot_v21(source_root: Path, output_root: Path, *, repo_id: str, overwrite: bool = False) -> Path:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    info_path = source_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Expected LeRobot export metadata at {info_path}")

    source_info = _load_json(info_path)
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 source export, got {source_info.get('codebase_version')!r}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}. Use overwrite=True to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    fps = int(source_info["fps"])
    features = source_info["features"]
    video_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    scalar_features = [key for key, feature in features.items() if feature["dtype"] != "video"]

    data = _read_parquet_tree(source_root / "data")
    episodes = _read_parquet_tree(source_root / "meta/episodes")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    tasks = _tasks_from_episodes(episodes)
    task_to_index = {task: index for index, task in enumerate(tasks)}

    total_episodes = int(len(episodes))
    total_frames = int(len(data))
    chunks_size = DEFAULT_CHUNK_SIZE
    info = {
        "codebase_version": "v2.1",
        "robot_type": source_info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": math.ceil(total_episodes / chunks_size),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": V21_DATA_PATH,
        "video_path": V21_VIDEO_PATH if video_keys else None,
        "features": features,
    }

    _write_json(output_root / "meta/info.json", info)
    _write_jsonl(output_root / "meta/tasks.jsonl", [{"task_index": index, "task": task} for index, task in enumerate(tasks)])

    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    all_stats: list[dict[str, dict[str, list[Any]]]] = []

    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        episode_chunk = _episode_chunk(episode_index, chunks_size)
        episode_data = data[data["episode_index"] == episode_index].copy()
        if len(episode_data) != int(episode["length"]):
            raise ValueError(
                f"Episode {episode_index} length mismatch: metadata={episode['length']} data={len(episode_data)}"
            )

        # v2.1 stores one parquet per episode and excludes video columns from parquet features.
        output_data_path = output_root / V21_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        episode_data[scalar_features].to_parquet(output_data_path, index=False)

        episode_tasks = list(episode["tasks"]) if not isinstance(episode["tasks"], str) else [episode["tasks"]]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": episode_tasks,
                "length": int(episode["length"]),
            }
        )

        stats = _scalar_feature_stats(episode_data, scalar_features)
        all_stats.append(stats)
        episode_stats_rows.append({"episode_index": episode_index, "stats": _to_jsonable(stats)})

    _write_jsonl(output_root / "meta/episodes.jsonl", episode_rows)
    _write_jsonl(output_root / "meta/episodes_stats.jsonl", episode_stats_rows)

    # v2.1 primarily reads per-episode stats, but keeping global stats helps with older tooling.
    global_stats = _scalar_feature_stats(data, scalar_features)
    _write_json(output_root / "meta/stats.json", _to_jsonable(global_stats))

    _write_episode_videos(source_root, output_root, episodes, video_keys, fps)
    print(f"OpenPI-compatible LeRobot v2.1 export complete: {output_root}")
    print(f"Repo id metadata: {repo_id}")
    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Orbit LeRobot v3 export root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination LeRobot v2.1 root.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id used by OpenPI.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output root if it exists.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    export_openpi_lerobot_v21(args.source_root, args.output_root, repo_id=args.repo_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
