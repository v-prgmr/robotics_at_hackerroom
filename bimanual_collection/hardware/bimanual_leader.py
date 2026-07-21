"""Bimanual leader abstraction for two SO-100 leader arms."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class LeaderArm(Protocol):
    """Minimal protocol implemented by LeRobot SO leader arms."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self, calibrate: bool = True) -> None: ...

    def get_action(self) -> dict[str, float]: ...

    def disconnect(self) -> None: ...


@dataclass(frozen=True)
class BimanualLeaderState:
    """Synchronized state read from both leader arms."""

    left: dict[str, float]
    right: dict[str, float]
    read_started_monotonic_s: float
    read_finished_monotonic_s: float


@dataclass(frozen=True)
class BimanualLeaderConfig:
    """Configuration for constructing two LeRobot SO leader arms."""

    left_port: str
    right_port: str
    left_id: str = "left_leader"
    right_id: str = "right_leader"
    calibration_dir: Path | None = None
    use_degrees: bool = True
    communication_timeout_s: float = 0.25


class BimanualLeader:
    """Wraps left and right leader arms with strict paired reads.

    The teleoperation loop must read both leader states before either follower is
    commanded. This class exposes a single `read()` method that either returns
    both sides or raises, allowing the caller to fail safe.
    """

    def __init__(
        self,
        config: BimanualLeaderConfig,
        left_arm: LeaderArm | None = None,
        right_arm: LeaderArm | None = None,
    ) -> None:
        self.config = config
        self.left_arm = left_arm
        self.right_arm = right_arm

    @classmethod
    def from_lerobot(cls, config: BimanualLeaderConfig) -> "BimanualLeader":
        """Create SO-100 leaders using LeRobot 0.4.4 APIs."""

        from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

        left = SO100Leader(
            SO100LeaderConfig(
                id=config.left_id,
                port=config.left_port,
                calibration_dir=config.calibration_dir,
                use_degrees=config.use_degrees,
            )
        )
        right = SO100Leader(
            SO100LeaderConfig(
                id=config.right_id,
                port=config.right_port,
                calibration_dir=config.calibration_dir,
                use_degrees=config.use_degrees,
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
            raise RuntimeError("Leader arms were not constructed")
        self.left_arm.connect(calibrate=calibrate)
        try:
            self.right_arm.connect(calibrate=calibrate)
        except Exception:
            self.disconnect()
            raise
        logger.info("Bimanual leaders connected")

    def read(self) -> BimanualLeaderState:
        """Read both leader states or raise on timeout/failure."""

        if self.left_arm is None or self.right_arm is None or not self.is_connected:
            raise RuntimeError("Bimanual leaders are not connected")

        start = time.monotonic()
        left = self.left_arm.get_action()
        right = self.right_arm.get_action()
        end = time.monotonic()

        if end - start > self.config.communication_timeout_s:
            raise TimeoutError(
                f"Leader read exceeded timeout: {end - start:.4f}s > {self.config.communication_timeout_s:.4f}s"
            )

        return BimanualLeaderState(
            left=left,
            right=right,
            read_started_monotonic_s=start,
            read_finished_monotonic_s=end,
        )

    def disconnect(self) -> None:
        for name, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if arm is None:
                continue
            try:
                if arm.is_connected:
                    arm.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting %s leader: %s", name, exc)
