"""Episode and timestep schemas for bimanual data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bimanual_collection.hardware.cameras import MatchedCameraFrame


@dataclass(frozen=True)
class EpisodeMetadata:
    """Human and system metadata saved with one episode."""

    episode_id: str
    task_description: str
    started_wall_time: str
    success: bool | None = None
    operator_notes: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    robot_calibration: dict[str, Any] = field(default_factory=dict)
    camera_calibration: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimestepSample:
    """One synchronized bimanual robot/control timestep."""

    episode_id: str
    timestep_index: int
    monotonic_timestamp_s: float
    wall_timestamp_s: float
    left_leader_joints: dict[str, float]
    right_leader_joints: dict[str, float]
    left_follower_joints: dict[str, float]
    right_follower_joints: dict[str, float]
    left_gripper_state: float | None
    right_gripper_state: float | None
    left_commanded_action: dict[str, float]
    right_commanded_action: dict[str, float]
    camera_matches: dict[str, MatchedCameraFrame]
    measured_control_hz: float
    loop_duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


def vector_from_joints(joints: dict[str, float]) -> list[float]:
    """Return joint values in stable insertion order."""

    return [float(value) for value in joints.values()]


def gripper_state(joints: dict[str, float]) -> float | None:
    """Extract the SO gripper state when present."""

    for key in ("gripper.pos", "left_gripper.pos", "right_gripper.pos", "gripper"):
        if key in joints:
            return float(joints[key])
    return None
