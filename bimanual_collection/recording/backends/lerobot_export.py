"""Deterministic converter from the intermediate format to LeRobot v3.0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def _read_video_frames(path: Path) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    return frames


def export_to_lerobot(intermediate_root: Path, output_root: Path, repo_id: str, fps: int) -> Any:
    """Convert selected-frame intermediate episodes to LeRobotDataset v3.0.

    This produces one LeRobot image/video frame per robot timestep using the
    timestamp index. Duplicate camera frames in the intermediate stream are
    duplicated in the LeRobot output so standard policy training code can index
    observations by timestep without consulting sidecars.
    """

    from lerobot.datasets.lerobot_dataset import LeRobotDataset


    episodes = sorted(p for p in intermediate_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not episodes:
        raise ValueError(f"No intermediate episodes found: {intermediate_root}")

    first_df = pd.read_parquet(episodes[0] / "timesteps.parquet")
    first_row = first_df.iloc[0]
    state_dim = len(first_row["left_follower_joints"]) + len(first_row["right_follower_joints"])
    action_dim = len(first_row["left_commanded_action"]) + len(first_row["right_commanded_action"])
    camera_names = sorted({col.removesuffix("_video_frame_index") for col in first_df.columns if col.endswith("_video_frame_index")})

    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    for camera in camera_names:
        video_path = episodes[0] / f"videos/{camera}.mp4"
        frame_cache = _read_video_frames(video_path)
        if not frame_cache:
            continue
        frame = next(iter(frame_cache.values()))
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": frame.shape,
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_root,
        robot_type="bi_so_follower",
        features=features,
        use_videos=True,
        streaming_encoding=True,
    )

    for episode in episodes:
        df = pd.read_parquet(episode / "timesteps.parquet")
        metadata_path = episode / "episode_metadata.json"
        task = ""
        if metadata_path.exists():
            import json

            with metadata_path.open("r", encoding="utf-8") as file:
                task = json.load(file).get("task_description", "")
        caches = {camera: _read_video_frames(episode / f"videos/{camera}.mp4") for camera in camera_names}
        first_ts = float(df["monotonic_timestamp_s"].iloc[0])
        for _, row in df.iterrows():
            frame = {
                "timestamp": float(row["monotonic_timestamp_s"] - first_ts),
                "observation.state": np.asarray(row["left_follower_joints"] + row["right_follower_joints"], dtype=np.float32),
                "action": np.asarray(row["left_commanded_action"] + row["right_commanded_action"], dtype=np.float32),
                "task": task,
            }
            for camera in camera_names:
                video_index = row.get(f"{camera}_video_frame_index")
                if bool(pd.isna(video_index)):
                    raise ValueError(f"Missing frame for camera {camera} in {episode}")
                frame[f"observation.images.{camera}"] = caches[camera][int(video_index)]
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return dataset
