"""Episode recorder independent of hardware and teleoperation."""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bimanual_collection.recording.backends.intermediate import IntermediateBackend, IntermediateBackendConfig
from bimanual_collection.recording.episode import EpisodeMetadata, TimestepSample


@dataclass(frozen=True)
class RecorderConfig:
    """Configures episode recording and warm-up behavior."""

    output_dir: Path
    warmup_s: float = 0.0
    task_description: str = ""
    camera_fps: int = 30
    episode_start_number: int | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    robot_calibration: dict[str, Any] = field(default_factory=dict)
    camera_calibration: dict[str, Any] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    """Lifecycle wrapper for start/record/stop/discard/save."""

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        backend_cfg = IntermediateBackendConfig(
            output_dir=config.output_dir,
            camera_fps=config.camera_fps,
            dataset_metadata=config.dataset_metadata,
        )
        self.backend = IntermediateBackend(backend_cfg)
        self._writer = None
        self._started_monotonic_s: float | None = None
        self._episode_id: str | None = None
        self._episode_number: int | None = None
        self._sample_count = 0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def episode_number(self) -> int | None:
        return self._episode_number

    @property
    def episode_label(self) -> str:
        if self._episode_number is not None:
            return f"Episode {self._episode_number}"
        return self._episode_id or "Episode"

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def elapsed_s(self) -> float:
        if self._started_monotonic_s is None:
            return 0.0
        return time.monotonic() - self._started_monotonic_s

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    @property
    def current_final_dir(self) -> Path | None:
        if self._episode_id is None:
            return None
        return self.config.output_dir / self._episode_id

    def _episode_path_in_use(self, number: int) -> bool:
        episode_id = f"episode-{number:06d}"
        if (self.config.output_dir / episode_id).exists():
            return True
        return any(self.config.output_dir.glob(f".{episode_id}.tmp-*"))

    def _next_episode_number(self) -> int:
        if self.config.episode_start_number is not None:
            number = self.config.episode_start_number
            while self._episode_path_in_use(number):
                number += 1
            return number

        pattern = re.compile(r"^episode-(\d+)$")
        max_number = 0
        for path in self.config.output_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                max_number = max(max_number, int(match.group(1)))
        return max_number + 1

    def start(self, task_description: str | None = None, episode_id: str | None = None) -> str:
        if self._writer is not None:
            raise RuntimeError("An episode is already recording")
        if episode_id is None:
            self._episode_number = self._next_episode_number()
            self._episode_id = f"episode-{self._episode_number:06d}"
        else:
            self._episode_id = episode_id
            match = re.fullmatch(r"episode-(\d+)", episode_id)
            self._episode_number = int(match.group(1)) if match else None
        self._started_monotonic_s = time.monotonic()
        self._sample_count = 0
        metadata = EpisodeMetadata(
            episode_id=self._episode_id,
            task_description=task_description or self.config.task_description,
            started_wall_time=datetime.now(timezone.utc).isoformat(),
            environment=self.config.environment,
            robot_calibration=self.config.robot_calibration,
            camera_calibration=self.config.camera_calibration,
        )
        self._writer = self.backend.start_episode(metadata)
        return self._episode_id

    def add_sample(self, sample: TimestepSample) -> None:
        if self._writer is None or self._started_monotonic_s is None:
            return
        if sample.monotonic_timestamp_s - self._started_monotonic_s < self.config.warmup_s:
            return
        self._writer.add_sample(sample)
        self._sample_count += 1

    def add_event(self, event: dict[str, Any]) -> None:
        """Add a control event to the current episode sidecar."""

        if self._writer is None:
            return
        self._writer.add_event(event)

    def stop_and_save(self, success: bool | None = None, operator_notes: str = "") -> Path:
        if self._writer is None:
            raise RuntimeError("No episode is recording")
        path = self._writer.save(success=success, operator_notes=operator_notes)
        self._writer = None
        self._started_monotonic_s = None
        self._episode_id = None
        self._episode_number = None
        self._sample_count = 0
        return path

    def discard(self) -> None:
        if self._writer is None:
            return
        self._writer.discard()
        self._writer = None
        self._started_monotonic_s = None
        self._episode_id = None
        self._episode_number = None
        self._sample_count = 0
