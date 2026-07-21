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


def _read_first_video_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read first frame from video: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class SequentialVideoFrameReader:
    """Read frames by index without caching whole videos in memory."""

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


def _joint_vector(*parts: Any) -> np.ndarray:
    """Concatenate left/right joint vectors without NumPy elementwise addition."""

    return np.concatenate([np.asarray(part, dtype=np.float32) for part in parts]).astype(np.float32, copy=False)


def _build_lerobot_frame(row: Any, task: str) -> dict[str, Any]:
    """Build scalar/vector LeRobot frame fields accepted by LeRobot 0.4.4."""

    return {
        "observation.state": _joint_vector(row["left_follower_joints"], row["right_follower_joints"]),
        "action": _joint_vector(row["left_commanded_action"], row["right_commanded_action"]),
        "task": task,
    }


def export_to_lerobot(
    intermediate_root: Path,
    output_root: Path,
    repo_id: str,
    fps: int,
    *,
    vcodec: str = "h264",
    encoder_threads: int | None = 1,
) -> Any:
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
        frame = _read_first_video_frame(video_path)
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
        vcodec=vcodec,
        encoder_threads=encoder_threads,
    )

    for episode_index, episode in enumerate(episodes, start=1):
        print(f"Exporting episode {episode_index}/{len(episodes)}: {episode.name}")
        df = pd.read_parquet(episode / "timesteps.parquet")
        metadata_path = episode / "episode_metadata.json"
        task = ""
        if metadata_path.exists():
            import json

            with metadata_path.open("r", encoding="utf-8") as file:
                task = json.load(file).get("task_description", "")
        readers = {camera: SequentialVideoFrameReader(episode / f"videos/{camera}.mp4") for camera in camera_names}
        try:
            for _, row in df.iterrows():
                frame = _build_lerobot_frame(row, task)
                for camera in camera_names:
                    video_index = row.get(f"{camera}_video_frame_index")
                    if video_index is None or bool(pd.isna(video_index)):
                        raise ValueError(f"Missing frame for camera {camera} in {episode}")
                    frame[f"observation.images.{camera}"] = readers[camera].get(int(video_index))
                dataset.add_frame(frame)
            dataset.save_episode()
        finally:
            for reader in readers.values():
                reader.close()
    dataset.finalize()
    return dataset
