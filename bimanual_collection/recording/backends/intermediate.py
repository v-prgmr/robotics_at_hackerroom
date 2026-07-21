"""Atomic intermediate dataset backend with Parquet metadata and MP4 video."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bimanual_collection.recording.episode import EpisodeMetadata, TimestepSample, vector_from_joints
from bimanual_collection.recording.video_writer import VideoWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntermediateBackendConfig:
    """Configuration for the default recording backend."""

    output_dir: Path
    camera_fps: int = 30
    video_codec: str = "libx264"
    video_crf: int = 23
    dataset_metadata: dict[str, Any] = field(default_factory=dict)


class IntermediateEpisodeWriter:
    """Writes one episode into a temporary directory before atomic publish."""

    def __init__(self, config: IntermediateBackendConfig, metadata: EpisodeMetadata) -> None:
        self.config = config
        self.metadata = metadata
        self.final_dir = config.output_dir / metadata.episode_id
        self.tmp_dir = config.output_dir / f".{metadata.episode_id}.tmp-{int(time.time() * 1e6)}"
        self.rows: list[dict[str, Any]] = []
        self.camera_index_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self._video_writers: dict[str, VideoWriter] = {}
        self._camera_frame_to_video_index: dict[tuple[str, int], int] = {}
        self._closed = False
        self.tmp_dir.mkdir(parents=True, exist_ok=False)
        (self.tmp_dir / "videos").mkdir(parents=True, exist_ok=True)

    def add_sample(self, sample: TimestepSample) -> None:
        if self._closed:
            raise RuntimeError("Cannot add samples to a closed episode writer")

        row = {
            "episode_id": sample.episode_id,
            "timestep_index": sample.timestep_index,
            "monotonic_timestamp_s": sample.monotonic_timestamp_s,
            "wall_timestamp_s": sample.wall_timestamp_s,
            "left_leader_joints": vector_from_joints(sample.left_leader_joints),
            "right_leader_joints": vector_from_joints(sample.right_leader_joints),
            "left_follower_joints": vector_from_joints(sample.left_follower_joints),
            "right_follower_joints": vector_from_joints(sample.right_follower_joints),
            "left_leader_joint_names": list(sample.left_leader_joints),
            "right_leader_joint_names": list(sample.right_leader_joints),
            "left_follower_joint_names": list(sample.left_follower_joints),
            "right_follower_joint_names": list(sample.right_follower_joints),
            "left_gripper_state": sample.left_gripper_state,
            "right_gripper_state": sample.right_gripper_state,
            "left_commanded_action": vector_from_joints(sample.left_commanded_action),
            "right_commanded_action": vector_from_joints(sample.right_commanded_action),
            "left_action_names": list(sample.left_commanded_action),
            "right_action_names": list(sample.right_commanded_action),
            "measured_control_hz": sample.measured_control_hz,
            "loop_duration_s": sample.loop_duration_s,
            "metadata_json": json.dumps(sample.metadata, sort_keys=True),
        }
        for camera_name, match in sample.camera_matches.items():
            video_frame_index = None
            video_path = None
            if match.frame is not None and match.frame_index is not None:
                video_frame_index = self._append_unique_camera_frame(camera_name, match.frame)
                video_path = f"videos/{camera_name}.mp4"
            row[f"{camera_name}_video_frame_index"] = video_frame_index
            row[f"{camera_name}_camera_frame_index"] = match.frame_index
            row[f"{camera_name}_camera_timestamp_s"] = match.camera_timestamp_s
            row[f"{camera_name}_camera_host_timestamp_s"] = match.host_timestamp_s
            row[f"{camera_name}_camera_hardware_timestamp_s"] = match.hardware_timestamp_s
            row[f"{camera_name}_frame_age_s"] = match.age_s
            row[f"{camera_name}_dropped_frames"] = match.dropped_frames
            row[f"{camera_name}_stale"] = match.stale
            row[f"{camera_name}_missing"] = match.missing
            row[f"{camera_name}_disconnected"] = match.disconnected
            row[f"{camera_name}_video_path"] = video_path
            self.camera_index_rows.append(
                {
                    "episode_id": sample.episode_id,
                    "timestep_index": sample.timestep_index,
                    "camera_name": camera_name,
                    "video_path": video_path,
                    "video_frame_index": video_frame_index,
                    "camera_frame_index": match.frame_index,
                    "camera_timestamp_s": match.camera_timestamp_s,
                    "camera_host_timestamp_s": match.host_timestamp_s,
                    "camera_hardware_timestamp_s": match.hardware_timestamp_s,
                    "frame_age_s": match.age_s,
                    "stale": match.stale,
                    "missing": match.missing,
                    "dropped_frames": match.dropped_frames,
                    "disconnected": match.disconnected,
                }
            )
        self.rows.append(row)

    def add_event(self, event: dict[str, Any]) -> None:
        """Record a control-state transition sidecar event."""

        if self._closed:
            raise RuntimeError("Cannot add events to a closed episode writer")
        self.event_rows.append(event)

    def _append_unique_camera_frame(self, camera_name: str, frame) -> int:
        key = (camera_name, frame.frame_index)
        if key in self._camera_frame_to_video_index:
            return self._camera_frame_to_video_index[key]
        writer = self._video_writers.get(camera_name)
        if writer is None:
            writer = VideoWriter(
                self.tmp_dir / "videos" / f"{camera_name}.mp4",
                fps=self.config.camera_fps,
                codec=self.config.video_codec,
                crf=self.config.video_crf,
            )
            self._video_writers[camera_name] = writer
        video_index = writer.append(frame.image)
        self._camera_frame_to_video_index[key] = video_index
        return video_index

    def save(self, success: bool | None = None, operator_notes: str | None = None) -> Path:
        """Close files and atomically publish the episode directory."""

        if self.final_dir.exists():
            raise FileExistsError(f"Episode already exists: {self.final_dir}")
        for writer in self._video_writers.values():
            writer.close()

        metadata = asdict(self.metadata)
        if success is not None:
            metadata["success"] = success
        if operator_notes is not None:
            metadata["operator_notes"] = operator_notes
        metadata["num_timesteps"] = len(self.rows)
        metadata["camera_video_frames"] = {name: writer.frame_count for name, writer in self._video_writers.items()}

        with (self.tmp_dir / "episode_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        pd.DataFrame(self.rows).to_parquet(self.tmp_dir / "timesteps.parquet", index=False)
        pd.DataFrame(self.camera_index_rows).to_parquet(self.tmp_dir / "camera_index.parquet", index=False)
        pd.DataFrame(self.event_rows).to_parquet(self.tmp_dir / "control_events.parquet", index=False)
        self._write_dataset_metadata()
        self.tmp_dir.rename(self.final_dir)
        self._closed = True
        logger.info("Saved episode %s", self.final_dir)
        return self.final_dir

    def _write_dataset_metadata(self) -> None:
        metadata_path = self.config.output_dir / "dataset_metadata.json"
        if metadata_path.exists():
            return
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(self.config.dataset_metadata, file, indent=2, sort_keys=True)

    def discard(self) -> None:
        """Delete temporary episode files."""

        for writer in self._video_writers.values():
            writer.close()
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self._closed = True
        logger.info("Discarded episode %s", self.metadata.episode_id)


class IntermediateBackend:
    """Factory for atomic intermediate episode writers."""

    def __init__(self, config: IntermediateBackendConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(self, metadata: EpisodeMetadata) -> IntermediateEpisodeWriter:
        return IntermediateEpisodeWriter(self.config, metadata)
