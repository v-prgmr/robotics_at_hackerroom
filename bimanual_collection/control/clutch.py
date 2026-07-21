"""Relative bimanual clutch and pause controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ArmSide = Literal["left", "right"]
StartupAlignment = Literal["hold_current", "leader_absolute"]


@dataclass(frozen=True)
class ControlEvent:
    """State transition emitted by clutch/pause control."""

    event: str
    side: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ControlOutput:
    """Target actions and recording gates for one control cycle."""

    left_action: dict[str, float]
    right_action: dict[str, float]
    should_record: bool
    left_clutch_active: bool
    right_clutch_active: bool
    recording_paused: bool
    events: list[ControlEvent] = field(default_factory=list)


def _copy_pose(pose: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in pose.items() if key.endswith(".pos") or key == "gripper"}


class ArmRelativeController:
    """Relative controller for one leader/follower pair with clutch support."""

    def __init__(self, side: ArmSide, startup_alignment: StartupAlignment = "hold_current") -> None:
        self.side = side
        self.startup_alignment = startup_alignment
        self.leader_reference: dict[str, float] | None = None
        self.follower_reference: dict[str, float] | None = None
        self.last_commanded_action: dict[str, float] | None = None
        self.clutch_active = False

    @property
    def is_initialized(self) -> bool:
        return (
            self.leader_reference is not None
            and self.follower_reference is not None
            and self.last_commanded_action is not None
        )

    def reset_references(self, leader_pose: dict[str, float], follower_pose: dict[str, float]) -> None:
        """Set references so the next target equals the current follower pose."""

        self.leader_reference = _copy_pose(leader_pose)
        self.follower_reference = _copy_pose(follower_pose)
        self.last_commanded_action = dict(self.follower_reference)

    def initialize_references(self, leader_pose: dict[str, float], follower_pose: dict[str, float]) -> None:
        """Initialize startup references according to the configured alignment mode."""

        if self.startup_alignment == "leader_absolute":
            leader = _copy_pose(leader_pose)
            self.leader_reference = dict(leader)
            self.follower_reference = dict(leader)
            self.last_commanded_action = dict(leader)
            return
        self.reset_references(leader_pose, follower_pose)

    def update(
        self,
        leader_pose: dict[str, float],
        follower_pose: dict[str, float],
        clutch_requested: bool,
    ) -> tuple[dict[str, float], list[ControlEvent]]:
        if not self.is_initialized:
            self.initialize_references(leader_pose, follower_pose)

        events: list[ControlEvent] = []
        if clutch_requested and not self.clutch_active:
            self.clutch_active = True
            events.append(ControlEvent(event="clutch_engaged", side=self.side))
        elif not clutch_requested and self.clutch_active:
            self.clutch_active = False
            self.reset_references(leader_pose, follower_pose)
            events.append(ControlEvent(event="clutch_released", side=self.side))

        if self.clutch_active:
            assert self.last_commanded_action is not None
            return dict(self.last_commanded_action), events

        assert self.leader_reference is not None
        assert self.follower_reference is not None
        target: dict[str, float] = {}
        for key, leader_value in _copy_pose(leader_pose).items():
            if key not in self.leader_reference or key not in self.follower_reference:
                continue
            target[key] = self.follower_reference[key] + (float(leader_value) - self.leader_reference[key])
        self.last_commanded_action = target
        return dict(target), events


class BimanualClutchController:
    """Coordinates two independent arm clutch controllers and pause state."""

    def __init__(self, startup_alignment: StartupAlignment = "hold_current") -> None:
        if startup_alignment not in ("hold_current", "leader_absolute"):
            raise ValueError(f"Unsupported startup alignment: {startup_alignment}")
        self.startup_alignment = startup_alignment
        self.left = ArmRelativeController("left", startup_alignment=startup_alignment)
        self.right = ArmRelativeController("right", startup_alignment=startup_alignment)
        self.recording_paused = False

    def reset_all_references(
        self,
        left_leader: dict[str, float],
        right_leader: dict[str, float],
        left_follower: dict[str, float],
        right_follower: dict[str, float],
    ) -> None:
        self.left.reset_references(left_leader, left_follower)
        self.right.reset_references(right_leader, right_follower)

    def update(
        self,
        left_leader: dict[str, float],
        right_leader: dict[str, float],
        left_follower: dict[str, float],
        right_follower: dict[str, float],
        left_clutch_active: bool,
        right_clutch_active: bool,
        recording_paused: bool,
    ) -> ControlOutput:
        events: list[ControlEvent] = []
        if recording_paused != self.recording_paused:
            self.recording_paused = recording_paused
            event_name = "recording_paused" if recording_paused else "recording_resumed"
            events.append(ControlEvent(event=event_name))
            if not recording_paused:
                self.reset_all_references(left_leader, right_leader, left_follower, right_follower)

        if self.recording_paused:
            left_hold = self.left.last_commanded_action or _copy_pose(left_follower)
            right_hold = self.right.last_commanded_action or _copy_pose(right_follower)
            return ControlOutput(
                left_action=dict(left_hold),
                right_action=dict(right_hold),
                should_record=False,
                left_clutch_active=left_clutch_active,
                right_clutch_active=right_clutch_active,
                recording_paused=True,
                events=events,
            )

        left_action, left_events = self.left.update(left_leader, left_follower, left_clutch_active)
        right_action, right_events = self.right.update(right_leader, right_follower, right_clutch_active)
        events.extend(left_events)
        events.extend(right_events)
        return ControlOutput(
            left_action=left_action,
            right_action=right_action,
            should_record=not left_clutch_active and not right_clutch_active,
            left_clutch_active=left_clutch_active,
            right_clutch_active=right_clutch_active,
            recording_paused=False,
            events=events,
        )
