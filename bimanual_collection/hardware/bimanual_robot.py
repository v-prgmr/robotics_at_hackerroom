"""Bimanual follower abstraction for two SO-100 follower arms."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class FollowerArm(Protocol):
    """Minimal protocol implemented by LeRobot SO follower arms."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self, calibrate: bool = True) -> None: ...

    def get_observation(self) -> dict[str, float]: ...

    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class JointLimit:
    """Inclusive scalar limit for a normalized LeRobot motor command."""

    minimum: float
    maximum: float

    def clip(self, value: float) -> float:
        return min(max(value, self.minimum), self.maximum)


@dataclass(frozen=True)
class BimanualRobotConfig:
    """Configuration for constructing two LeRobot SO follower arms."""

    left_port: str
    right_port: str
    left_id: str = "left_follower"
    right_id: str = "right_follower"
    calibration_dir: Path | None = None
    use_degrees: bool = True
    disable_torque_on_disconnect: bool = True
    communication_timeout_s: float = 0.25
    max_relative_target: float | dict[str, float] | None = None
    joint_limits: dict[str, JointLimit] = field(default_factory=dict)


@dataclass(frozen=True)
class BimanualFollowerState:
    """Synchronized state read from both follower arms."""

    left: dict[str, float]
    right: dict[str, float]
    read_started_monotonic_s: float
    read_finished_monotonic_s: float


@dataclass(frozen=True)
class BimanualCommandResult:
    """Actions actually accepted by the followers after clipping and safety checks."""

    left: dict[str, float]
    right: dict[str, float]
    command_started_monotonic_s: float
    command_finished_monotonic_s: float


class BimanualRobot:
    """Wraps left and right SO-100 followers with fail-safe paired control."""

    def __init__(
        self,
        config: BimanualRobotConfig,
        left_arm: FollowerArm | None = None,
        right_arm: FollowerArm | None = None,
    ) -> None:
        self.config = config
        self.left_arm = left_arm
        self.right_arm = right_arm
        self._last_state: BimanualFollowerState | None = None

    @classmethod
    def from_lerobot(cls, config: BimanualRobotConfig) -> "BimanualRobot":
        """Create SO-100 followers using LeRobot 0.4.4 APIs."""

        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        left = SO100Follower(
            SO100FollowerConfig(
                id=config.left_id,
                port=config.left_port,
                calibration_dir=config.calibration_dir,
                use_degrees=config.use_degrees,
                cameras={},
                disable_torque_on_disconnect=config.disable_torque_on_disconnect,
                max_relative_target=config.max_relative_target,
            )
        )
        right = SO100Follower(
            SO100FollowerConfig(
                id=config.right_id,
                port=config.right_port,
                calibration_dir=config.calibration_dir,
                use_degrees=config.use_degrees,
                cameras={},
                disable_torque_on_disconnect=config.disable_torque_on_disconnect,
                max_relative_target=config.max_relative_target,
            )
        )
        return cls(config=config, left_arm=left, right_arm=right)

    @property
    def is_connected(self) -> bool:
        return bool(
            self.left_arm
            and self.right_arm
            and self.left_arm.is_connected
            and self.right_arm.is_connected
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.left_arm is None or self.right_arm is None:
            raise RuntimeError("Follower arms were not constructed")
        self.left_arm.connect(calibrate=calibrate)
        try:
            self.right_arm.connect(calibrate=calibrate)
        except Exception:
            self.disconnect()
            raise
        logger.info("Bimanual followers connected")

    def read_state(self) -> BimanualFollowerState:
        """Read both follower observations or raise on timeout/failure."""

        if self.left_arm is None or self.right_arm is None or not self.is_connected:
            raise RuntimeError("Bimanual followers are not connected")

        start = time.monotonic()
        left = self.left_arm.get_observation()
        right = self.right_arm.get_observation()
        end = time.monotonic()
        if end - start > self.config.communication_timeout_s:
            raise TimeoutError(
                f"Follower read exceeded timeout: {end - start:.4f}s > {self.config.communication_timeout_s:.4f}s"
            )
        state = BimanualFollowerState(left=left, right=right, read_started_monotonic_s=start, read_finished_monotonic_s=end)
        self._last_state = state
        return state

    def send_actions(self, left_action: dict[str, float], right_action: dict[str, float]) -> BimanualCommandResult:
        """Send both follower commands after validating and clipping both sides.

        If either command fails, both arms are disconnected so one arm cannot keep
        moving uncontrolled while the other side has failed.
        """

        if self.left_arm is None or self.right_arm is None or not self.is_connected:
            raise RuntimeError("Bimanual followers are not connected")

        left_clipped = self.clip_action(left_action, side="left")
        right_clipped = self.clip_action(right_action, side="right")
        start = time.monotonic()
        try:
            sent_left = self.left_arm.send_action(left_clipped)
            sent_right = self.right_arm.send_action(right_clipped)
        except Exception:
            logger.exception("Follower command failed; disconnecting both arms")
            self.disconnect()
            raise
        end = time.monotonic()
        if end - start > self.config.communication_timeout_s:
            self.disconnect()
            raise TimeoutError(
                f"Follower command exceeded timeout: {end - start:.4f}s > {self.config.communication_timeout_s:.4f}s"
            )
        return BimanualCommandResult(sent_left, sent_right, start, end)

    def move_to_positions(
        self,
        left_target: dict[str, float],
        right_target: dict[str, float],
        *,
        duration_s: float = 2.0,
        steps: int = 120,
    ) -> None:
        """Move both followers from their current pose to target poses by linear interpolation."""

        if steps < 1:
            steps = 1
        start_state = self.read_state()
        left_start = {key: float(value) for key, value in start_state.left.items() if key in left_target}
        right_start = {key: float(value) for key, value in start_state.right.items() if key in right_target}
        if set(left_start) != set(left_target):
            missing = sorted(set(left_target) - set(left_start))
            raise ValueError(f"Left follower home target contains unknown joints: {missing}")
        if set(right_start) != set(right_target):
            missing = sorted(set(right_target) - set(right_start))
            raise ValueError(f"Right follower home target contains unknown joints: {missing}")

        period_s = duration_s / steps if duration_s > 0 else 0.0
        for step in range(1, steps + 1):
            alpha = step / steps
            left = {key: left_start[key] + (float(left_target[key]) - left_start[key]) * alpha for key in left_target}
            right = {key: right_start[key] + (float(right_target[key]) - right_start[key]) * alpha for key in right_target}
            self.send_actions(left, right)
            if period_s > 0 and step < steps:
                time.sleep(period_s)

    def clip_action(self, action: dict[str, float], side: str) -> dict[str, float]:
        """Apply finite-value checks and configured joint limits."""

        clipped: dict[str, float] = {}
        for key, value in action.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {side} action value for {key}: {value!r}")
            limit = self.config.joint_limits.get(key) or self.config.joint_limits.get(f"{side}.{key}")
            clipped[key] = limit.clip(float(value)) if limit else float(value)
        return clipped

    def hold_position(self) -> None:
        """Command the last observed state if available."""

        if self._last_state is None:
            return
        self.send_actions(
            {k: v for k, v in self._last_state.left.items() if k in DEFAULT_JOINT_NAMES or k.endswith(".pos")},
            {k: v for k, v in self._last_state.right.items() if k in DEFAULT_JOINT_NAMES or k.endswith(".pos")},
        )

    def disconnect(self) -> None:
        for name, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if arm is None:
                continue
            try:
                if arm.is_connected:
                    arm.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting %s follower: %s", name, exc)
