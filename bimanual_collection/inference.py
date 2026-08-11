"""Async VLA policy deployment for bimanual SO-100 follower arms."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from bimanual_collection.bimanual_teleop import (
    build_camera_configs,
    calibration_id,
    cfg_value,
    default_home_position_file,
    load_follower_home_positions,
    load_config,
    move_followers_home,
    parse_joint_limits,
    print_camera_preflight,
    setup_logging,
    teleop_value,
)
from bimanual_collection.hardware.bimanual_robot import (
    DEFAULT_JOINT_NAMES,
    BimanualFollowerState,
    BimanualRobot,
    BimanualRobotConfig,
)
from bimanual_collection.hardware.cameras import CameraConfigError, MatchedCameraFrame, MultiCameraManager
from bimanual_collection.recording.episode import TimestepSample, gripper_state
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig

logger = logging.getLogger(__name__)


class DeploymentState:
    IDLE = "IDLE"
    ARMED_WAITING_FOR_CHUNK = "ARMED_WAITING_FOR_CHUNK"
    RUNNING = "RUNNING"
    HOLDING_STALE_POLICY = "HOLDING_STALE_POLICY"
    HOMING = "HOMING"
    ESTOP = "ESTOP"


class ChunkExecutionMode:
    RECEDING = "receding"
    FULL = "full"


CHUNK_EXECUTION_MODES = (ChunkExecutionMode.RECEDING, ChunkExecutionMode.FULL)
DEFAULT_POLICY_ACTION_HZ = 16.57
DEFAULT_ACTION_BRIDGE_DURATION_S = 0.1

ROLLOUT_SUCCESS = "success"
ROLLOUT_FAILURE = "failure"


@dataclass(frozen=True)
class ObservationSnapshot:
    sequence: int
    timestamp_s: float
    follower_state: BimanualFollowerState
    camera_matches: dict[str, MatchedCameraFrame]


@dataclass(frozen=True)
class TimedAction:
    action: np.ndarray
    source_timestamp_s: float
    published_timestamp_s: float | None
    chunk_id: int | None
    action_index: int
    upper_action_index: int | None = None
    interpolation_alpha: float = 0.0
    phase: str = "trajectory"
    trajectory_elapsed_s: float | None = None
    request_id: int = 0


@dataclass(frozen=True)
class PoppedAction:
    action: TimedAction
    queue_remaining: int


@dataclass(frozen=True)
class ActionRequestToken:
    generation: int
    request_id: int


@dataclass(frozen=True)
class RerunTelemetryConfig:
    """Runtime telemetry destination and throttling for live Rerun visualization."""

    application_id: str = "orbit_bimanual_inference"
    spawn: bool = True
    connect_grpc_url: str | None = None
    save_path: Path | None = None
    camera_fps: float = 10.0
    max_queue: int = 512
    policy_action_hz: float = DEFAULT_POLICY_ACTION_HZ


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DebugTraceWriter:
    """Thread-safe debug trace for diagnosing live policy deployment failures."""

    def __init__(self, root: Path, metadata: dict[str, Any] | None = None) -> None:
        timestamp = datetime.now(timezone.utc).strftime("run-%Y%m%d_%H%M%S_%f")
        self.run_dir = root.expanduser() / timestamp
        self.images_dir = self.run_dir / "images"
        self.action_chunks_dir = self.run_dir / "action_chunks"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self.action_chunks_dir.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self._events_file = (self.run_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata or {}, file, indent=2, sort_keys=True, default=_json_default)

    def close(self) -> None:
        with self._lock:
            self._events_file.close()

    def log_event(self, event: str, **payload: Any) -> None:
        row = {
            "event": event,
            "monotonic_timestamp_s": time.monotonic(),
            "wall_timestamp_s": time.time(),
            **payload,
        }
        line = json.dumps(row, sort_keys=True, default=_json_default)
        with self._lock:
            if not self._events_file.closed:
                self._events_file.write(line + "\n")

    def log_policy_observation(
        self,
        snapshot: ObservationSnapshot,
        *,
        image_feature_keys: list[str],
        joint_names: tuple[str, ...],
    ) -> None:
        camera_rows = []
        for image_key in image_feature_keys:
            camera_name = image_key.removeprefix("observation.images.")
            match = snapshot.camera_matches.get(camera_name)
            image_path = None
            if match is not None and match.frame is not None:
                image_path = self.images_dir / f"obs_{snapshot.sequence:06d}_{camera_name}.png"
                self._write_rgb_image(image_path, match.frame.image)
            camera_rows.append(
                {
                    "camera_name": camera_name,
                    "image_key": image_key,
                    "image_path": str(image_path.relative_to(self.run_dir)) if image_path is not None else None,
                    "frame_index": None if match is None else match.frame_index,
                    "camera_timestamp_s": None if match is None else match.camera_timestamp_s,
                    "host_timestamp_s": None if match is None else match.host_timestamp_s,
                    "age_s": None if match is None else match.age_s,
                    "stale": True if match is None else match.stale,
                    "missing": True if match is None else match.missing,
                    "disconnected": True if match is None else match.disconnected,
                }
            )
        self.log_event(
            "policy_observation",
            observation_sequence=snapshot.sequence,
            observation_timestamp_s=snapshot.timestamp_s,
            left_follower=snapshot.follower_state.left,
            right_follower=snapshot.follower_state.right,
            observation_state=bimanual_joint_vector(
                snapshot.follower_state.left,
                snapshot.follower_state.right,
                joint_names,
            ),
            cameras=camera_rows,
        )

    def log_policy_chunk(
        self,
        snapshot: ObservationSnapshot,
        *,
        chunk_id: int,
        normalized_actions: np.ndarray | None,
        postprocessed_actions: np.ndarray,
        inference_duration_s: float,
    ) -> None:
        post_path = self.action_chunks_dir / f"chunk_{chunk_id:06d}_obs_{snapshot.sequence:06d}_postprocessed.npy"
        np.save(post_path, postprocessed_actions)
        normalized_path = None
        if normalized_actions is not None:
            normalized_path = self.action_chunks_dir / f"chunk_{chunk_id:06d}_obs_{snapshot.sequence:06d}_normalized.npy"
            np.save(normalized_path, normalized_actions)
        self.log_event(
            "policy_chunk",
            chunk_id=chunk_id,
            observation_sequence=snapshot.sequence,
            observation_timestamp_s=snapshot.timestamp_s,
            inference_duration_s=inference_duration_s,
            postprocessed_action_chunk_path=str(post_path.relative_to(self.run_dir)),
            normalized_action_chunk_path=str(normalized_path.relative_to(self.run_dir))
            if normalized_path is not None
            else None,
            postprocessed_action_shape=list(postprocessed_actions.shape),
            first_postprocessed_action=postprocessed_actions[0] if len(postprocessed_actions) else [],
        )

    def log_robot_action(
        self,
        *,
        action: np.ndarray,
        left_action: dict[str, float],
        right_action: dict[str, float],
        timed_action: TimedAction,
        queue_remaining: int,
        dry_run: bool,
    ) -> None:
        self.log_event(
            "robot_action",
            action=action,
            left_action=left_action,
            right_action=right_action,
            source_timestamp_s=timed_action.source_timestamp_s,
            published_timestamp_s=timed_action.published_timestamp_s,
            chunk_id=timed_action.chunk_id,
            action_index=timed_action.action_index,
            upper_action_index=timed_action.upper_action_index,
            interpolation_alpha=timed_action.interpolation_alpha,
            phase=timed_action.phase,
            trajectory_elapsed_s=timed_action.trajectory_elapsed_s,
            request_id=timed_action.request_id,
            queue_remaining=queue_remaining,
            action_age_s=time.monotonic() - timed_action.source_timestamp_s,
            dry_run=dry_run,
        )

    def log_hold(self, *, reason: str, armed: bool) -> None:
        self.log_event("hold", reason=reason, armed=armed)

    def log_state_transition(self, *, state: str, reason: str) -> None:
        self.log_event("state_transition", state=state, reason=reason)

    @staticmethod
    def _write_rgb_image(path: Path, image: np.ndarray) -> None:
        import cv2  # type: ignore

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _rerun_entity_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


class RerunLiveVisualizer:
    """Nonblocking Rerun telemetry writer for live policy execution.

    The robot and policy threads only enqueue small events. Rerun logging happens
    on a daemon thread so a slow or disconnected viewer cannot stall control.
    """

    def __init__(
        self,
        config: RerunTelemetryConfig,
        *,
        joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES,
    ) -> None:
        self.config = config
        self.joint_names = tuple(joint_names)
        self._events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(
            maxsize=max(1, config.max_queue)
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rr: Any | None = None
        self._last_camera_log_s = 0.0
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def start(self) -> None:
        if self._thread is not None:
            return
        import rerun as rr  # type: ignore

        self._rr = rr
        rr.init(
            self.config.application_id,
            spawn=bool(self.config.spawn),
        )
        if self.config.connect_grpc_url is not None:
            rr.connect_grpc(self.config.connect_grpc_url)
        if self.config.save_path is not None:
            self.config.save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(self.config.save_path)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rerun-live-visualizer", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._enqueue_sentinel()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def log_observation(
        self,
        snapshot: ObservationSnapshot,
        *,
        image_feature_keys: list[str],
        joint_names: tuple[str, ...],
    ) -> None:
        now_s = time.monotonic()
        camera_period_s = 1.0 / max(self.config.camera_fps, 0.001)
        include_images = now_s - self._last_camera_log_s >= camera_period_s
        if include_images:
            self._last_camera_log_s = now_s

        cameras = []
        for image_key in image_feature_keys:
            camera_name = image_key.removeprefix("observation.images.")
            match = snapshot.camera_matches.get(camera_name)
            image = None
            if include_images and match is not None and match.frame is not None:
                image = np.ascontiguousarray(match.frame.image).copy()
            cameras.append(
                {
                    "name": camera_name,
                    "image": image,
                    "age_s": None if match is None else match.age_s,
                    "stale": True if match is None else match.stale,
                    "missing": True if match is None else match.missing,
                    "disconnected": True if match is None else match.disconnected,
                    "dropped_frames": 0 if match is None else match.dropped_frames,
                }
            )

        self._try_enqueue(
            "observation",
            {
                "sequence": snapshot.sequence,
                "timestamp_s": snapshot.timestamp_s,
                "left": dict(snapshot.follower_state.left),
                "right": dict(snapshot.follower_state.right),
                "joint_names": tuple(joint_names),
                "cameras": cameras,
            },
        )

    def log_policy_chunk(
        self,
        snapshot: ObservationSnapshot,
        *,
        chunk_id: int,
        postprocessed_actions: np.ndarray,
        inference_duration_s: float,
    ) -> None:
        self._try_enqueue(
            "policy_chunk",
            {
                "sequence": snapshot.sequence,
                "timestamp_s": snapshot.timestamp_s,
                "chunk_id": chunk_id,
                "actions": np.asarray(postprocessed_actions, dtype=np.float32).copy(),
                "inference_duration_s": float(inference_duration_s),
            },
        )

    def log_robot_action(
        self,
        *,
        action: np.ndarray,
        left_action: dict[str, float],
        right_action: dict[str, float],
        timed_action: TimedAction,
        queue_remaining: int,
        dry_run: bool,
    ) -> None:
        self._try_enqueue(
            "robot_action",
            {
                "timestamp_s": time.monotonic(),
                "action": np.asarray(action, dtype=np.float32).copy(),
                "left_action": dict(left_action),
                "right_action": dict(right_action),
                "source_timestamp_s": timed_action.source_timestamp_s,
                "chunk_id": timed_action.chunk_id,
                "action_index": timed_action.action_index,
                "upper_action_index": timed_action.upper_action_index,
                "interpolation_alpha": timed_action.interpolation_alpha,
                "phase": timed_action.phase,
                "trajectory_elapsed_s": timed_action.trajectory_elapsed_s,
                "request_id": timed_action.request_id,
                "queue_remaining": int(queue_remaining),
                "action_age_s": time.monotonic() - timed_action.source_timestamp_s,
                "dry_run": bool(dry_run),
            },
        )

    def log_hold(self, *, reason: str, armed: bool) -> None:
        self._try_enqueue("hold", {"timestamp_s": time.monotonic(), "reason": reason, "armed": bool(armed)})

    def log_state_transition(self, *, state: str, reason: str) -> None:
        self._try_enqueue(
            "state_transition",
            {"timestamp_s": time.monotonic(), "state": state, "reason": reason},
        )

    def _enqueue_sentinel(self) -> None:
        try:
            self._events.put_nowait(None)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._events.get_nowait()
            with contextlib.suppress(queue.Full):
                self._events.put_nowait(None)

    def _try_enqueue(self, event: str, payload: dict[str, Any]) -> None:
        if self._stop.is_set():
            return
        try:
            self._events.put_nowait((event, payload))
        except queue.Full:
            self._dropped_events += 1

    def _run(self) -> None:
        while True:
            item = self._events.get()
            if item is None:
                return
            event, payload = item
            try:
                self._log_event(event, payload)
            except Exception:
                logger.exception("Failed to log Rerun telemetry event: %s", event)

    def _log_event(self, event: str, payload: dict[str, Any]) -> None:
        if self._rr is None:
            return
        handlers = {
            "observation": self._log_observation_event,
            "policy_chunk": self._log_policy_chunk_event,
            "robot_action": self._log_robot_action_event,
            "hold": self._log_hold_event,
            "state_transition": self._log_state_transition_event,
        }
        handlers[event](payload)

    def _set_time(self, *, timestamp_s: float | None = None, sequence: int | None = None) -> None:
        assert self._rr is not None
        if timestamp_s is not None:
            self._rr.set_time_seconds("monotonic_time", float(timestamp_s))
        if sequence is not None:
            self._rr.set_time_sequence("observation_sequence", int(sequence))

    def _log_scalar(self, path: str, value: Any) -> None:
        assert self._rr is not None
        if value is None:
            return
        self._rr.log(path, self._rr.Scalars(float(value)))

    def _log_observation_event(self, payload: dict[str, Any]) -> None:
        assert self._rr is not None
        self._set_time(timestamp_s=payload["timestamp_s"], sequence=payload["sequence"])
        self._log_joint_dict("joints/left_follower", payload["left"], payload["joint_names"])
        self._log_joint_dict("joints/right_follower", payload["right"], payload["joint_names"])
        for camera in payload["cameras"]:
            camera_name = _rerun_entity_name(camera["name"])
            if camera["image"] is not None:
                self._rr.log(f"cameras/{camera_name}/image", self._rr.Image(camera["image"]))
            base_path = f"diagnostics/cameras/{camera_name}"
            self._log_scalar(f"{base_path}/age_s", camera["age_s"])
            self._log_scalar(f"{base_path}/stale", int(camera["stale"]))
            self._log_scalar(f"{base_path}/missing", int(camera["missing"]))
            self._log_scalar(f"{base_path}/disconnected", int(camera["disconnected"]))
            self._log_scalar(f"{base_path}/dropped_frames", camera["dropped_frames"])

    def _log_policy_chunk_event(self, payload: dict[str, Any]) -> None:
        assert self._rr is not None
        self._set_time(timestamp_s=payload["timestamp_s"], sequence=payload["sequence"])
        self._rr.set_time_sequence("policy_chunk_id", int(payload["chunk_id"]))
        actions = np.asarray(payload["actions"], dtype=np.float32)
        self._log_scalar("diagnostics/policy/inference_duration_s", payload["inference_duration_s"])
        self._log_scalar("diagnostics/policy/chunk_length", len(actions))
        if actions.ndim != 2:
            return
        steps = np.arange(actions.shape[0], dtype=np.float32) / self.config.policy_action_hz
        for side, offset in (("left", 0), ("right", len(self.joint_names))):
            for joint_index, joint_name in enumerate(self.joint_names):
                dim = offset + joint_index
                if dim >= actions.shape[1]:
                    continue
                values = actions[:, dim].astype(np.float32, copy=False)
                strip = np.column_stack([steps, values])
                entity = f"policy_chunk/{side}/{_rerun_entity_name(joint_name)}"
                self._rr.log(entity, self._rr.LineStrips2D([strip]))
                if len(values):
                    entity = f"policy_chunk_first_action/{side}/{_rerun_entity_name(joint_name)}"
                    self._log_scalar(entity, values[0])

    def _log_robot_action_event(self, payload: dict[str, Any]) -> None:
        self._set_time(timestamp_s=payload["timestamp_s"])
        self._log_joint_dict("commands/left_follower", payload["left_action"], self.joint_names)
        self._log_joint_dict("commands/right_follower", payload["right_action"], self.joint_names)
        self._log_scalar("diagnostics/action/age_s", payload["action_age_s"])
        self._log_scalar("diagnostics/action/queue_remaining", payload["queue_remaining"])
        self._log_scalar("diagnostics/action/chunk_id", payload["chunk_id"])
        self._log_scalar("diagnostics/action/action_index", payload["action_index"])
        self._log_scalar("diagnostics/action/upper_action_index", payload["upper_action_index"])
        self._log_scalar("diagnostics/action/interpolation_alpha", payload["interpolation_alpha"])
        self._log_scalar("diagnostics/action/trajectory_elapsed_s", payload["trajectory_elapsed_s"])
        self._log_scalar("diagnostics/action/request_id", payload["request_id"])
        self._log_scalar(
            "diagnostics/action/phase",
            {"request_hold": 0, "bridge": 1, "trajectory": 2}.get(payload["phase"], -1),
        )
        self._log_scalar("diagnostics/action/dry_run", int(payload["dry_run"]))

    def _log_hold_event(self, payload: dict[str, Any]) -> None:
        assert self._rr is not None
        self._set_time(timestamp_s=payload["timestamp_s"])
        self._rr.log("events/holds", self._rr.TextLog(payload["reason"], level="WARN"))
        self._log_scalar("diagnostics/holding/armed", int(payload["armed"]))

    def _log_state_transition_event(self, payload: dict[str, Any]) -> None:
        assert self._rr is not None
        self._set_time(timestamp_s=payload["timestamp_s"])
        reason = f": {payload['reason']}" if payload.get("reason") else ""
        self._rr.log("events/state", self._rr.TextLog(f"{payload['state']}{reason}", level="INFO"))

    def _log_joint_dict(self, base_path: str, joints: dict[str, float], joint_names: tuple[str, ...]) -> None:
        for joint_name in joint_names:
            value = joints.get(joint_name)
            if value is None:
                continue
            self._log_scalar(f"{base_path}/{_rerun_entity_name(joint_name)}", value)


class StopFlag:
    """Signal-safe stop flag used for Ctrl+C and emergency-stop paths."""

    def __init__(self) -> None:
        self.stop = False
        self.reason = ""

    def request(self, reason: str) -> None:
        self.stop = True
        self.reason = reason


class DeploymentHotkeys:
    """Thread-safe edge flags set by keyboard callbacks and consumed by the robot loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arm_toggle = False
        self._reset = False
        self._home = False
        self._classification: str | None = None

    def request_arm_toggle(self) -> None:
        with self._lock:
            self._arm_toggle = True

    def request_reset(self) -> None:
        with self._lock:
            self._reset = True

    def request_home(self) -> None:
        with self._lock:
            self._home = True

    def request_classification(self, classification: str) -> None:
        if classification not in {ROLLOUT_SUCCESS, ROLLOUT_FAILURE}:
            raise ValueError(f"Unknown rollout classification: {classification}")
        with self._lock:
            if self._classification is None:
                self._classification = classification

    def consume_arm_toggle(self) -> bool:
        with self._lock:
            requested = self._arm_toggle
            self._arm_toggle = False
        return requested

    def consume_reset(self) -> bool:
        with self._lock:
            requested = self._reset
            self._reset = False
        return requested

    def consume_home(self) -> bool:
        with self._lock:
            requested = self._home
            self._home = False
        return requested

    def consume_classification(self) -> str | None:
        with self._lock:
            classification = self._classification
            self._classification = None
        return classification


class StateTransitionLogger:
    """Print deployment state transitions once per state change."""

    def __init__(
        self,
        debug_trace: DebugTraceWriter | None = None,
        live_visualizer: RerunLiveVisualizer | None = None,
    ) -> None:
        self.state: str | None = None
        self.debug_trace = debug_trace
        self.live_visualizer = live_visualizer

    def set(self, state: str, reason: str = "") -> None:
        if state == self.state:
            return
        self.state = state
        suffix = f": {reason}" if reason else ""
        print(f"State -> {state}{suffix}")
        logger.info("Deployment state -> %s%s", state, suffix)
        if self.debug_trace is not None:
            self.debug_trace.log_state_transition(state=state, reason=reason)
        if self.live_visualizer is not None:
            self.live_visualizer.log_state_transition(state=state, reason=reason)


class LatestObservationBuffer:
    """Single-slot observation handoff from the robot loop to the policy worker."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: ObservationSnapshot | None = None
        self._history: deque[ObservationSnapshot] = deque(maxlen=8)
        self._sequence = 0

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._history.clear()
            self._condition.notify_all()

    def publish(
        self,
        *,
        timestamp_s: float,
        follower_state: BimanualFollowerState,
        camera_matches: dict[str, MatchedCameraFrame],
    ) -> int:
        with self._condition:
            self._sequence += 1
            self._latest = ObservationSnapshot(
                sequence=self._sequence,
                timestamp_s=timestamp_s,
                follower_state=follower_state,
                camera_matches=camera_matches,
            )
            self._history.append(self._latest)
            self._condition.notify_all()
            return self._sequence

    def wait_for_new(self, last_sequence: int, timeout_s: float) -> ObservationSnapshot | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._latest is None or self._latest.sequence <= last_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._latest

    def history_ending_at(
        self,
        snapshot: ObservationSnapshot,
        *,
        steps: int,
        period_s: float,
    ) -> list[ObservationSnapshot]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        with self._condition:
            candidates = [item for item in self._history if item.sequence <= snapshot.sequence]
        if not candidates:
            return [snapshot] * steps
        targets = [snapshot.timestamp_s - period_s * offset for offset in reversed(range(steps))]
        return [min(candidates, key=lambda item: abs(item.timestamp_s - target)) for target in targets]


class ActionChunkBuffer:
    """Time-sampled policy trajectory with a fixed hold during inference."""

    def __init__(
        self,
        *,
        policy_action_hz: float = DEFAULT_POLICY_ACTION_HZ,
        bridge_duration_s: float = DEFAULT_ACTION_BRIDGE_DURATION_S,
    ) -> None:
        if not np.isfinite(policy_action_hz) or policy_action_hz <= 0:
            raise ValueError("policy_action_hz must be > 0")
        if not np.isfinite(bridge_duration_s) or bridge_duration_s < 0:
            raise ValueError("bridge_duration_s must be >= 0")
        self._lock = threading.Lock()
        self.policy_action_hz = float(policy_action_hz)
        self.bridge_duration_s = float(bridge_duration_s)
        self._published_chunks = 0
        self._current_chunk_id = 0
        self._generation = 0
        self._request_id = 0
        self._request_pending = False
        self._hold_action: np.ndarray | None = None
        self._source_timestamp_s = 0.0
        self._actions: np.ndarray | None = None
        self._published_timestamp_s: float | None = None
        self._trajectory_start_s: float | None = None
        self._bridge_start_action: np.ndarray | None = None

    @property
    def published_chunks(self) -> int:
        with self._lock:
            return self._published_chunks

    def clear(self) -> None:
        with self._lock:
            self._clear_locked(invalidate=True)

    def _clear_locked(self, *, invalidate: bool) -> None:
        if invalidate:
            self._generation += 1
        self._request_pending = False
        self._hold_action = None
        self._actions = None
        self._published_timestamp_s = None
        self._trajectory_start_s = None
        self._bridge_start_action = None

    def should_request_chunk(
        self,
        execution_horizon: int,
        replan_threshold: int,
        chunk_execution_mode: str,
        now_s: float | None = None,
    ) -> bool:
        with self._lock:
            if self._request_pending or (self._actions is not None and self._trajectory_start_s is None):
                return False
            if self._actions is None:
                return True
            position = self._trajectory_position_locked(time.monotonic() if now_s is None else now_s)
            if chunk_execution_mode == ChunkExecutionMode.FULL:
                return position + 1e-9 >= len(self._actions)
            consumed = min(int(np.floor(max(position, 0.0))) + 1, len(self._actions))
            remaining = len(self._actions) - consumed
            return remaining <= replan_threshold or consumed >= execution_horizon

    def queue_length(self) -> int:
        with self._lock:
            if self._actions is None:
                return 0
            if self._trajectory_start_s is None:
                return len(self._actions)
            consumed = min(
                int(np.floor(max(self._trajectory_position_locked(time.monotonic()), 0.0))) + 1,
                len(self._actions),
            )
            return len(self._actions) - consumed

    def begin_request(self, *, source_timestamp_s: float, hold_action: np.ndarray) -> ActionRequestToken | None:
        hold = np.asarray(hold_action, dtype=np.float32)
        if hold.ndim != 1 or not np.all(np.isfinite(hold)):
            raise ValueError(f"Expected finite hold action shape (D,), got {hold.shape}")
        with self._lock:
            if self._request_pending:
                return None
            self._request_id += 1
            self._request_pending = True
            self._hold_action = hold.copy()
            self._source_timestamp_s = float(source_timestamp_s)
            self._actions = None
            self._published_timestamp_s = None
            self._trajectory_start_s = None
            self._bridge_start_action = None
            return ActionRequestToken(self._generation, self._request_id)

    def cancel_request(self, token: ActionRequestToken) -> None:
        with self._lock:
            if token == ActionRequestToken(self._generation, self._request_id):
                self._clear_locked(invalidate=True)

    def publish_chunk(self, token: ActionRequestToken, actions: np.ndarray) -> int | None:
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or len(chunk) < 1:
            raise ValueError(f"Expected nonempty action chunk shape (T, D), got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("Action chunk contains non-finite values")
        with self._lock:
            if token != ActionRequestToken(self._generation, self._request_id) or not self._request_pending:
                return None
            if self._hold_action is None or chunk.shape[1] != len(self._hold_action):
                expected = None if self._hold_action is None else len(self._hold_action)
                raise ValueError(f"Expected action dimension {expected}, got {chunk.shape[1]}")
            self._current_chunk_id += 1
            chunk_id = self._current_chunk_id
            self._request_pending = False
            self._actions = chunk.copy()
            self._published_timestamp_s = time.monotonic()
            self._trajectory_start_s = None
            self._bridge_start_action = None
            self._published_chunks += 1
            return chunk_id

    def sample_timed_action(
        self,
        now_s: float,
        measured_action: np.ndarray,
        max_age_s: float | None,
    ) -> PoppedAction | None:
        measured = np.asarray(measured_action, dtype=np.float32)
        with self._lock:
            if self._request_pending:
                assert self._hold_action is not None
                return PoppedAction(
                    TimedAction(
                        action=self._hold_action.copy(),
                        source_timestamp_s=self._source_timestamp_s,
                        published_timestamp_s=None,
                        chunk_id=None,
                        action_index=0,
                        phase="request_hold",
                        request_id=self._request_id,
                    ),
                    queue_remaining=0,
                )
            if self._actions is None:
                return None
            if measured.shape != (self._actions.shape[1],):
                raise ValueError(f"Expected measured action shape {(self._actions.shape[1],)}, got {measured.shape}")
            if self._trajectory_start_s is None:
                # Reject an old response before it starts. Once accepted, the
                # trajectory's deliberately slower knot timing may exceed this age.
                if max_age_s is not None and now_s - self._source_timestamp_s > max_age_s:
                    self._clear_locked(invalidate=True)
                    return None
                self._trajectory_start_s = now_s
                self._bridge_start_action = measured.copy()

            assert self._trajectory_start_s is not None
            assert self._bridge_start_action is not None
            elapsed_s = max(0.0, now_s - self._trajectory_start_s)
            if self.bridge_duration_s > 0 and elapsed_s < self.bridge_duration_s:
                alpha = elapsed_s / self.bridge_duration_s
                target = (1.0 - alpha) * self._bridge_start_action + alpha * self._actions[0]
                phase = "bridge"
                action_index = 0
                upper_action_index = 0
                trajectory_elapsed_s = 0.0
                remaining = len(self._actions)
            else:
                trajectory_elapsed_s = max(0.0, elapsed_s - self.bridge_duration_s)
                position = trajectory_elapsed_s * self.policy_action_hz
                action_index = min(int(np.floor(position)), len(self._actions) - 1)
                upper_action_index = min(action_index + 1, len(self._actions) - 1)
                alpha = min(max(position - action_index, 0.0), 1.0)
                target = (1.0 - alpha) * self._actions[action_index] + alpha * self._actions[upper_action_index]
                phase = "trajectory"
                consumed = min(action_index + 1, len(self._actions))
                remaining = len(self._actions) - consumed

            return PoppedAction(
                TimedAction(
                    action=np.asarray(target, dtype=np.float32),
                    source_timestamp_s=self._source_timestamp_s,
                    published_timestamp_s=self._published_timestamp_s,
                    chunk_id=self._current_chunk_id,
                    action_index=action_index,
                    upper_action_index=upper_action_index,
                    interpolation_alpha=float(alpha),
                    phase=phase,
                    trajectory_elapsed_s=trajectory_elapsed_s,
                    request_id=self._request_id,
                ),
                queue_remaining=remaining,
            )

    def _trajectory_position_locked(self, now_s: float) -> float:
        if self._trajectory_start_s is None:
            return 0.0
        elapsed_s = max(0.0, now_s - self._trajectory_start_s - self.bridge_duration_s)
        return elapsed_s * self.policy_action_hz


class PolicyWorker:
    """Runs VLA inference off the realtime robot loop."""

    def __init__(
        self,
        *,
        runtime: "PolicyRuntime",
        observation_adapter: "ObservationAdapter",
        observation_buffer: LatestObservationBuffer,
        action_buffer: ActionChunkBuffer,
        armed: threading.Event,
        reset_requested: threading.Event,
        stop_requested: threading.Event,
        debug_trace: DebugTraceWriter | None = None,
        live_visualizer: RerunLiveVisualizer | None = None,
        execution_horizon: int = 10,
        replan_threshold: int = 5,
        chunk_execution_mode: str = ChunkExecutionMode.RECEDING,
        wait_timeout_s: float = 0.1,
        generation_provider: Callable[[], int] | None = None,
        response_accepted: Callable[[int], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.observation_adapter = observation_adapter
        self.observation_buffer = observation_buffer
        self.action_buffer = action_buffer
        self.armed = armed
        self.reset_requested = reset_requested
        self.stop_requested = stop_requested
        self.debug_trace = debug_trace
        self.live_visualizer = live_visualizer
        self.execution_horizon = execution_horizon
        self.replan_threshold = replan_threshold
        self.chunk_execution_mode = chunk_execution_mode
        self.wait_timeout_s = wait_timeout_s
        self.generation_provider = generation_provider
        self.response_accepted = response_accepted
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vla-policy-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _run(self) -> None:
        last_sequence = 0
        while not self.stop_requested.is_set():
            if not self.armed.is_set():
                if self.reset_requested.wait(timeout=0.05):
                    self.action_buffer.clear()
                    self.runtime.reset()
                    last_sequence = 0
                if self.reset_requested.is_set():
                    self.reset_requested.clear()
                continue

            if self.reset_requested.is_set():
                self.action_buffer.clear()
                self.runtime.reset()
                last_sequence = 0
                self.reset_requested.clear()

            if not self.action_buffer.should_request_chunk(
                self.execution_horizon,
                self.replan_threshold,
                self.chunk_execution_mode,
            ):
                time.sleep(0.005)
                continue

            snapshot = self.observation_buffer.wait_for_new(last_sequence, self.wait_timeout_s)
            if snapshot is None:
                continue
            last_sequence = snapshot.sequence
            request_token = self.action_buffer.begin_request(
                source_timestamp_s=snapshot.timestamp_s,
                hold_action=bimanual_joint_vector(
                    snapshot.follower_state.left,
                    snapshot.follower_state.right,
                    self.observation_adapter.joint_names,
                ),
            )
            if request_token is None:
                continue
            if self.reset_requested.is_set() or not self.armed.is_set():
                self.action_buffer.cancel_request(request_token)
                continue

            try:
                request_generation = self.generation_provider() if self.generation_provider is not None else 0
                inference_start_s = time.monotonic()
                n_obs_steps = max(1, int(getattr(self.runtime, "n_obs_steps", 1)))
                if hasattr(self.observation_adapter, "build_history"):
                    snapshots = self.observation_buffer.history_ending_at(
                        snapshot,
                        steps=n_obs_steps,
                        period_s=1.0 / self.action_buffer.policy_action_hz,
                    )
                    observation = self.observation_adapter.build_history(snapshots)
                else:
                    observation = self.observation_adapter.build(snapshot)
                normalized_actions = None
                if self.debug_trace is not None:
                    self.debug_trace.log_policy_observation(
                        snapshot,
                        image_feature_keys=self.observation_adapter.image_feature_keys,
                        joint_names=self.observation_adapter.joint_names,
                    )
                    normalized_actions, actions = self.runtime.predict_action_chunk_with_debug(observation)
                else:
                    actions = self.runtime.predict_action_chunk(observation)
                inference_duration_s = time.monotonic() - inference_start_s
                if self.reset_requested.is_set() or not self.armed.is_set():
                    self.action_buffer.cancel_request(request_token)
                    continue
                if self.response_accepted is not None and not self.response_accepted(request_generation):
                    self.action_buffer.cancel_request(request_token)
                    continue
                chunk_id = self.action_buffer.publish_chunk(request_token, actions)
                if chunk_id is None:
                    continue
                if self.debug_trace is not None:
                    self.debug_trace.log_policy_chunk(
                        snapshot,
                        chunk_id=chunk_id,
                        normalized_actions=normalized_actions,
                        postprocessed_actions=actions,
                        inference_duration_s=inference_duration_s,
                    )
                if self.live_visualizer is not None:
                    self.live_visualizer.log_policy_chunk(
                        snapshot,
                        chunk_id=chunk_id,
                        postprocessed_actions=actions,
                        inference_duration_s=inference_duration_s,
                    )
            except Exception as exc:
                self.action_buffer.cancel_request(request_token)
                logger.exception("Policy inference failed")
                self._set_error(str(exc))
                self.stop_requested.set()
                return


class PolicyRuntime:
    """Loads a LeRobot policy and its saved pre/postprocessors from a checkpoint."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.policy: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.image_feature_keys: list[str] = []

    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")

        # Registers SmolVLA's saved processor steps before loading processor JSON.
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        if self.device is not None:
            cfg.device = self.device
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(self.checkpoint, config=cfg)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )
        self.image_feature_keys = [
            key for key, feature in cfg.input_features.items() if feature.type == FeatureType.VISUAL
        ]
        logger.info("Loaded %s policy from %s", cfg.type, self.checkpoint)

    def reset(self) -> None:
        if self.policy is None:
            raise RuntimeError("Policy is not loaded")
        self.policy.reset()

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        _normalized_actions, postprocessed_actions = self.predict_action_chunk_with_debug(observation)
        return postprocessed_actions

    def predict_action_chunk_with_debug(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Policy runtime is not loaded")
        batch = self.preprocessor(observation)
        with torch.inference_mode():
            normalized_actions = self.policy.predict_action_chunk(batch)
        postprocessed_actions = self.postprocessor(normalized_actions)
        return _action_chunk_to_numpy(normalized_actions), _action_chunk_to_numpy(postprocessed_actions)


def _action_chunk_to_numpy(actions: Any) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions


class ObservationAdapter:
    """Converts live bimanual observations into the LeRobot training feature schema."""

    def __init__(self, *, task_description: str, image_feature_keys: list[str], joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES) -> None:
        self.task_description = task_description
        self.image_feature_keys = list(image_feature_keys)
        self.joint_names = tuple(joint_names)

    @property
    def expected_action_dim(self) -> int:
        return len(self.joint_names) * 2

    def build(self, snapshot: ObservationSnapshot) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "observation.state": torch.from_numpy(
                bimanual_joint_vector(snapshot.follower_state.left, snapshot.follower_state.right, self.joint_names)
            ),
            "task": self.task_description,
        }
        for image_key in self.image_feature_keys:
            camera_name = image_key.removeprefix("observation.images.")
            match = snapshot.camera_matches.get(camera_name)
            if match is None or match.frame is None:
                raise ValueError(f"Missing required camera frame: {camera_name}")
            if match.stale or match.missing or match.disconnected:
                raise ValueError(f"Invalid camera frame for {camera_name}: stale={match.stale} missing={match.missing} disconnected={match.disconnected}")
            observation[image_key] = image_to_policy_tensor(match.frame.image)
        return observation


def bimanual_joint_vector(
    left: dict[str, float], right: dict[str, float], joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
) -> np.ndarray:
    values: list[float] = []
    for side_name, joints in (("left", left), ("right", right)):
        missing = [name for name in joint_names if name not in joints]
        if missing:
            raise ValueError(f"Missing {side_name} follower joints: {missing}")
        values.extend(float(joints[name]) for name in joint_names)
    return np.asarray(values, dtype=np.float32)


def image_to_policy_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0


def split_bimanual_action(
    action: np.ndarray | torch.Tensor,
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES,
) -> tuple[dict[str, float], dict[str, float]]:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = len(joint_names) * 2
    if vector.shape[0] != expected:
        raise ValueError(f"Expected action dimension {expected}, got {vector.shape[0]}")
    left_values = vector[: len(joint_names)]
    right_values = vector[len(joint_names) :]
    left = {name: float(value) for name, value in zip(joint_names, left_values, strict=True)}
    right = {name: float(value) for name, value in zip(joint_names, right_values, strict=True)}
    return left, right


def validate_action_delta(
    left_action: dict[str, float],
    right_action: dict[str, float],
    follower_state: BimanualFollowerState,
    max_delta: float | None,
) -> None:
    if max_delta is None:
        return
    for side, action, state in (
        ("left", left_action, follower_state.left),
        ("right", right_action, follower_state.right),
    ):
        for joint, target in action.items():
            if joint not in state:
                continue
            delta = abs(float(target) - float(state[joint]))
            if delta > max_delta:
                raise ValueError(f"{side} {joint} action delta {delta:.4f} exceeds limit {max_delta:.4f}")


def camera_matches_valid(matches: dict[str, MatchedCameraFrame], image_feature_keys: list[str]) -> bool:
    for image_key in image_feature_keys:
        camera_name = image_key.removeprefix("observation.images.")
        match = matches.get(camera_name)
        if match is None or match.frame is None or match.stale or match.missing or match.disconnected:
            return False
    return True


def install_signal_handlers(flag: StopFlag) -> None:
    def _handler(signum, _frame) -> None:
        flag.request(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def install_keyboard_controls(
    flag: StopFlag,
    hotkeys: DeploymentHotkeys,
    *,
    arm_key: str,
    reset_key: str,
    home_key: str,
    emergency_stop_key: str,
    failure_key: str | None = None,
    success_key: str | None = None,
) -> Any | None:
    """Start a pynput listener for deployment controls."""

    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Keyboard listener unavailable: %s", exc)
        print(f"Keyboard listener unavailable: {exc}")
        return None

    pressed: set[str] = set()

    def normalize(key: Any) -> str:
        try:
            name = str(key.char)
            return "space" if name == " " else name
        except AttributeError:
            return str(key).replace("Key.", "")

    def on_press(key: Any) -> None:
        name = normalize(key)
        if name in pressed:
            return
        pressed.add(name)
        if name == emergency_stop_key:
            flag.request(f"emergency stop key '{emergency_stop_key}'")
        elif name == arm_key:
            hotkeys.request_arm_toggle()
        elif name == reset_key:
            hotkeys.request_reset()
        elif name == home_key:
            hotkeys.request_home()
        elif failure_key is not None and name == failure_key:
            hotkeys.request_classification(ROLLOUT_FAILURE)
        elif success_key is not None and name == success_key:
            hotkeys.request_classification(ROLLOUT_SUCCESS)

    def on_release(key: Any) -> None:
        pressed.discard(normalize(key))

    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        return listener
    except Exception as exc:
        logger.warning("Keyboard listener failed to start: %s", exc)
        print(f"Keyboard listener failed to start: {exc}")
        return None


def precise_sleep_until(deadline_s: float) -> None:
    remaining = deadline_s - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


@dataclass
class PeriodicCaptureGate:
    """Select controller ticks at a lower dataset cadence without catch-up bursts."""

    capture_hz: float
    _next_capture_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.capture_hz) or self.capture_hz <= 0:
            raise ValueError("capture_hz must be > 0")

    def reset(self) -> None:
        self._next_capture_timestamp_s = None

    def should_capture(self, timestamp_s: float) -> bool:
        period_s = 1.0 / self.capture_hz
        if self._next_capture_timestamp_s is None:
            self._next_capture_timestamp_s = timestamp_s + period_s
            return True
        if timestamp_s < self._next_capture_timestamp_s:
            return False
        elapsed_periods = int((timestamp_s - self._next_capture_timestamp_s) // period_s) + 1
        self._next_capture_timestamp_s += elapsed_periods * period_s
        return True


@dataclass
class AutonomousRolloutCapture:
    """Stages an autonomous rollout and publishes it only after operator classification."""

    recorder: EpisodeRecorder
    success_output_dir: Path
    failure_output_dir: Path
    policy_checkpoint: str | None
    capture_hz: float = DEFAULT_POLICY_ACTION_HZ
    _capture_gate: PeriodicCaptureGate = field(init=False)

    def __post_init__(self) -> None:
        self._capture_gate = PeriodicCaptureGate(self.capture_hz)

    def start(self) -> None:
        self._capture_gate.reset()
        self.recorder.start_deferred(
            extra={
                "episode_type": "autonomous_rollout",
                "policy_checkpoint": self.policy_checkpoint,
                "capture_hz": self.capture_hz,
            }
        )

    def should_record(self, timestamp_s: float) -> bool:
        return self._capture_gate.should_capture(timestamp_s)

    def classify(self, classification: str) -> Path:
        success = classification == ROLLOUT_SUCCESS
        terminal_timestamp = datetime.now(timezone.utc).isoformat()
        path = self.recorder.stop_and_save_to(
            self.success_output_dir if success else self.failure_output_dir,
            success=success,
            extra_metadata={
                "terminal_success": success,
                "termination_reason": f"operator_{classification}",
                "policy_checkpoint": self.policy_checkpoint,
                "terminal_timestamp": terminal_timestamp,
                "episode_type": "autonomous_rollout",
            },
        )
        self._capture_gate.reset()
        return path

    def discard(self) -> None:
        self.recorder.discard()
        self._capture_gate.reset()


def move_followers_home_now(robot: BimanualRobot, config: dict[str, Any], args: argparse.Namespace) -> None:
    home_file = Path(teleop_value(args, config, "home_position_file", default_home_position_file(config))).expanduser()
    left_home, right_home = load_follower_home_positions(home_file)
    duration_s = float(teleop_value(args, config, "home_move_duration_s", 2.0))
    steps = int(teleop_value(args, config, "home_move_steps", 120))
    logger.info("Moving followers home from %s over %.2fs", home_file, duration_s)
    robot.move_to_positions(left_home, right_home, duration_s=duration_s, steps=steps)


def run_control_loop(
    *,
    robot: BimanualRobot,
    cameras: MultiCameraManager,
    policy_worker: PolicyWorker,
    observation_buffer: LatestObservationBuffer,
    action_buffer: ActionChunkBuffer,
    armed: threading.Event,
    reset_requested: threading.Event,
    policy_stop_requested: threading.Event,
    hotkeys: DeploymentHotkeys,
    robot_fps: int,
    image_feature_keys: list[str],
    max_action_age_s: float | None,
    max_action_delta: float | None,
    home_callback: Callable[[], None],
    dry_run: bool,
    flag: StopFlag,
    debug_trace: DebugTraceWriter | None = None,
    live_visualizer: RerunLiveVisualizer | None = None,
    rollout_capture: AutonomousRolloutCapture | None = None,
) -> None:
    period_s = 1.0 / robot_fps
    states = StateTransitionLogger(debug_trace, live_visualizer)
    states.set(DeploymentState.IDLE, "press the arm key to start policy control")
    had_running_actions = False
    previous_loop_start = time.monotonic()

    def discard_active_capture() -> None:
        if rollout_capture is not None and rollout_capture.recorder.is_recording:
            rollout_capture.discard()

    def record_tick_and_maybe_classify(
        *,
        follower_state: BimanualFollowerState,
        camera_matches: dict[str, MatchedCameraFrame],
        left_action: dict[str, float],
        right_action: dict[str, float],
        action_source: str,
        loop_start: float,
        sample_timestamp: float,
        timed_action: TimedAction | None = None,
        queue_remaining: int | None = None,
        hold_reason: str | None = None,
    ) -> bool:
        nonlocal previous_loop_start, had_running_actions
        interval_s = loop_start - previous_loop_start
        measured_hz = 1.0 / interval_s if interval_s > 0 else 0.0
        loop_duration_s = time.monotonic() - loop_start
        classification = hotkeys.consume_classification()
        capture_current_tick = (
            rollout_capture is not None
            and rollout_capture.recorder.is_recording
            and (classification is not None or rollout_capture.should_record(sample_timestamp))
        )
        if capture_current_tick:
            assert rollout_capture is not None
            episode_id = rollout_capture.recorder.episode_id
            assert episode_id is not None
            rollout_capture.recorder.add_sample(
                TimestepSample(
                    episode_id=episode_id,
                    timestep_index=rollout_capture.recorder.sample_count,
                    monotonic_timestamp_s=sample_timestamp,
                    wall_timestamp_s=time.time(),
                    left_leader_joints={},
                    right_leader_joints={},
                    left_follower_joints=dict(follower_state.left),
                    right_follower_joints=dict(follower_state.right),
                    left_gripper_state=gripper_state(follower_state.left),
                    right_gripper_state=gripper_state(follower_state.right),
                    left_commanded_action=dict(left_action),
                    right_commanded_action=dict(right_action),
                    camera_matches=camera_matches,
                    measured_control_hz=measured_hz,
                    loop_duration_s=loop_duration_s,
                    metadata={
                        "action_source": action_source,
                        "chunk_id": timed_action.chunk_id if timed_action is not None else None,
                        "action_index": timed_action.action_index if timed_action is not None else None,
                        "upper_action_index": timed_action.upper_action_index if timed_action is not None else None,
                        "interpolation_alpha": timed_action.interpolation_alpha if timed_action is not None else None,
                        "action_phase": timed_action.phase if timed_action is not None else None,
                        "trajectory_elapsed_s": timed_action.trajectory_elapsed_s if timed_action is not None else None,
                        "request_id": timed_action.request_id if timed_action is not None else None,
                        "queue_remaining": queue_remaining,
                        "hold_reason": hold_reason,
                        "dry_run": dry_run,
                    },
                )
            )

        if classification is None:
            previous_loop_start = loop_start
            return False
        if rollout_capture is None or not rollout_capture.recorder.is_recording:
            previous_loop_start = loop_start
            return False

        try:
            path = rollout_capture.classify(classification)
            print(f"Saved {classification} rollout: {path}")
            state_reason = f"{classification} rollout saved; policy disarmed"
        except Exception as exc:
            flag.request(f"failed to save {classification} rollout: {exc}")
            logger.exception("Failed to save classified rollout")
            discard_active_capture()
            state_reason = f"{classification} rollout save failed; policy disarmed"
        armed.clear()
        action_buffer.clear()
        observation_buffer.clear()
        reset_requested.set()
        had_running_actions = False
        states.set(DeploymentState.IDLE, state_reason)
        previous_loop_start = loop_start
        return True

    def hold_command(follower_state: BimanualFollowerState) -> tuple[dict[str, float], dict[str, float]]:
        left = {
            k: v for k, v in follower_state.left.items() if k in DEFAULT_JOINT_NAMES or k.endswith(".pos")
        }
        right = {
            k: v for k, v in follower_state.right.items() if k in DEFAULT_JOINT_NAMES or k.endswith(".pos")
        }
        if dry_run:
            return robot.clip_action(left, side="left"), robot.clip_action(right, side="right")
        result = robot.hold_position()
        if result is None:
            raise RuntimeError("Cannot hold before a follower state has been read")
        return result.left, result.right

    while not flag.stop and not policy_stop_requested.is_set():
        loop_start = time.monotonic()

        if hotkeys.consume_arm_toggle():
            if armed.is_set():
                armed.clear()
                discard_active_capture()
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                had_running_actions = False
                states.set(DeploymentState.IDLE, "policy disarmed")
            else:
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                if rollout_capture is not None:
                    try:
                        rollout_capture.start()
                    except Exception as exc:
                        flag.request(f"failed to start rollout capture: {exc}")
                        logger.exception("Failed to start rollout capture")
                        break
                armed.set()
                had_running_actions = False
                states.set(DeploymentState.ARMED_WAITING_FOR_CHUNK, "policy armed")

        if hotkeys.consume_reset():
            discard_active_capture()
            action_buffer.clear()
            observation_buffer.clear()
            reset_requested.set()
            had_running_actions = False
            if armed.is_set():
                states.set(DeploymentState.ARMED_WAITING_FOR_CHUNK, "policy reset requested")
            else:
                states.set(DeploymentState.IDLE, "policy reset requested")

        if hotkeys.consume_home():
            armed.clear()
            discard_active_capture()
            action_buffer.clear()
            observation_buffer.clear()
            reset_requested.set()
            had_running_actions = False
            states.set(DeploymentState.HOMING, "policy disarmed; moving followers home")
            try:
                home_callback()
            except Exception as exc:
                flag.request(f"home failed: {exc}")
                logger.exception("Failed to move followers home")
                break
            observation_buffer.clear()
            reset_requested.set()
            states.set(DeploymentState.IDLE, "followers homed; press the arm key to start policy control")
            previous_loop_start = time.monotonic()
            continue

        worker_error = policy_worker.error
        if worker_error is not None:
            flag.request(f"policy failure: {worker_error}")
            break

        try:
            follower_state = robot.read_state()
            sample_timestamp = time.monotonic()
            camera_sample = cameras.match(sample_timestamp)
        except Exception as exc:
            flag.request(f"hardware failure: {exc}")
            logger.exception("Hardware read failed; stopping inference")
            break

        if not armed.is_set():
            hotkeys.consume_classification()
            if not dry_run:
                with contextlib.suppress(Exception):
                    robot.hold_position()
            precise_sleep_until(loop_start + period_s)
            previous_loop_start = loop_start
            continue

        camera_valid = camera_matches_valid(camera_sample.matches, image_feature_keys)
        if camera_valid:
            sequence = observation_buffer.publish(
                timestamp_s=sample_timestamp,
                follower_state=follower_state,
                camera_matches=camera_sample.matches,
            )
            if live_visualizer is not None:
                live_visualizer.log_observation(
                    ObservationSnapshot(
                        sequence=sequence,
                        timestamp_s=sample_timestamp,
                        follower_state=follower_state,
                        camera_matches=camera_sample.matches,
                    ),
                    image_feature_keys=image_feature_keys,
                    joint_names=DEFAULT_JOINT_NAMES,
                )
        else:
            observation_buffer.clear()
            action_buffer.clear()
            reset_requested.set()
            if debug_trace is not None:
                debug_trace.log_hold(reason="invalid camera frame", armed=True)
            if live_visualizer is not None:
                live_visualizer.log_hold(reason="invalid camera frame", armed=True)

        measured_action = bimanual_joint_vector(
            follower_state.left,
            follower_state.right,
            DEFAULT_JOINT_NAMES,
        )
        popped_action = action_buffer.sample_timed_action(
            time.monotonic(),
            measured_action,
            max_action_age_s,
        )
        if popped_action is None:
            hold_reason = "missing or stale policy action" if camera_valid else "invalid camera frame"
            state = (
                DeploymentState.HOLDING_STALE_POLICY
                if had_running_actions
                else DeploymentState.ARMED_WAITING_FOR_CHUNK
            )
            states.set(state, "holding current follower pose")
            if debug_trace is not None:
                debug_trace.log_hold(reason=hold_reason, armed=True)
            if live_visualizer is not None:
                live_visualizer.log_hold(reason=hold_reason, armed=True)
            try:
                left_action, right_action = hold_command(follower_state)
            except Exception as exc:
                flag.request(f"hold command failed: {exc}")
                logger.exception("Failed to command follower hold")
                break
            if record_tick_and_maybe_classify(
                follower_state=follower_state,
                camera_matches=camera_sample.matches,
                left_action=left_action,
                right_action=right_action,
                action_source="hold",
                hold_reason=hold_reason,
                loop_start=loop_start,
                sample_timestamp=sample_timestamp,
            ):
                continue
            precise_sleep_until(loop_start + period_s)
            continue

        try:
            timed_action = popped_action.action
            action = timed_action.action
            left_action, right_action = split_bimanual_action(action)
            validate_action_delta(left_action, right_action, follower_state, max_action_delta)
            if dry_run:
                left_action = robot.clip_action(left_action, side="left")
                right_action = robot.clip_action(right_action, side="right")
            else:
                command_result = robot.send_actions(left_action, right_action)
                left_action = command_result.left
                right_action = command_result.right
            if debug_trace is not None:
                debug_trace.log_robot_action(
                    action=action,
                    left_action=left_action,
                    right_action=right_action,
                    timed_action=timed_action,
                    queue_remaining=popped_action.queue_remaining,
                    dry_run=dry_run,
                )
            if live_visualizer is not None:
                live_visualizer.log_robot_action(
                    action=action,
                    left_action=left_action,
                    right_action=right_action,
                    timed_action=timed_action,
                    queue_remaining=popped_action.queue_remaining,
                    dry_run=dry_run,
                )
        except Exception as exc:
            flag.request(f"unsafe action: {exc}")
            logger.exception("Unsafe policy action; stopping inference")
            break

        if timed_action.phase != "request_hold":
            had_running_actions = True
        interval_s = loop_start - previous_loop_start
        measured_hz = 1.0 / interval_s if interval_s > 0 else 0.0
        if record_tick_and_maybe_classify(
            follower_state=follower_state,
            camera_matches=camera_sample.matches,
            left_action=left_action,
            right_action=right_action,
            action_source="request_hold" if timed_action.phase == "request_hold" else "policy",
            timed_action=timed_action,
            queue_remaining=popped_action.queue_remaining,
            loop_start=loop_start,
            sample_timestamp=sample_timestamp,
        ):
            continue
        if timed_action.phase == "request_hold":
            states.set(DeploymentState.ARMED_WAITING_FOR_CHUNK, "holding fixed request pose during inference")
        else:
            states.set(
                DeploymentState.RUNNING,
                f"robot commands {measured_hz:.1f} Hz; policy knots {action_buffer.policy_action_hz:.2f} Hz",
            )
        precise_sleep_until(loop_start + period_s)

    worker_error = policy_worker.error
    if worker_error is not None and not flag.reason:
        flag.request(f"policy failure: {worker_error}")
    states.set(DeploymentState.ESTOP, flag.reason or "policy worker stopped")
    armed.clear()
    action_buffer.clear()
    discard_active_capture()
    if not dry_run and robot.is_connected:
        with contextlib.suppress(Exception):
            robot.hold_position()


def build_arg_parser(*, checkpoint_required: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=checkpoint_required,
        help="Path to pretrained_model checkpoint directory",
    )
    parser.add_argument("--task-description")
    parser.add_argument("--device", help="Torch device override, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--left-robot-port")
    parser.add_argument("--right-robot-port")
    parser.add_argument("--overhead-camera")
    parser.add_argument("--left-wrist-camera")
    parser.add_argument("--right-wrist-camera")
    parser.add_argument("--robot-fps", type=int)
    parser.add_argument(
        "--policy-action-hz",
        type=float,
        help=f"Temporal rate of policy trajectory knots; defaults to {DEFAULT_POLICY_ACTION_HZ} Hz.",
    )
    parser.add_argument(
        "--action-bridge-duration-s",
        type=float,
        help=(
            "Time to interpolate from the response-time measured pose to action 0; "
            f"defaults to {DEFAULT_ACTION_BRIDGE_DURATION_S} s, use 0 to disable."
        ),
    )
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-buffer-size", type=int)
    parser.add_argument("--camera-stale-after-s", type=float)
    parser.add_argument("--camera-timeout-s", type=float)
    parser.add_argument("--move-followers-to-home", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--home-position-file", type=Path)
    parser.add_argument("--home-move-duration-s", type=float)
    parser.add_argument("--home-move-steps", type=int)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--communication-timeout-s", type=float)
    parser.add_argument("--max-action-age-s", type=float, default=1.0)
    parser.add_argument(
        "--disable-action-age-check",
        action="store_true",
        help="Do not reject a policy response when its source observation is too old to begin playback.",
    )
    parser.add_argument("--max-action-delta", type=float, help="Reject commands farther than this from current joint state")
    parser.add_argument("--policy-wait-timeout-s", type=float, default=0.1)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=10,
        help="In receding mode, request a new chunk after this many actions have been consumed.",
    )
    parser.add_argument(
        "--replan-threshold",
        type=int,
        default=5,
        help="In receding mode, request a new chunk when queued actions fall to this count.",
    )
    parser.add_argument(
        "--chunk-execution-mode",
        choices=CHUNK_EXECUTION_MODES,
        default=ChunkExecutionMode.FULL,
        help="Use 'receding' to replace chunks while executing, or 'full' to request the next chunk only after the current queue is empty.",
    )
    parser.add_argument("--debug-trace-dir", type=Path, help="Write inference debug trace under this directory")
    parser.add_argument(
        "--rerun-live",
        action="store_true",
        help="Open a live Rerun viewer for cameras, joints, actions, and policy chunks",
    )
    parser.add_argument(
        "--rerun-connect-grpc",
        help=(
            "Stream Rerun telemetry to an existing viewer, "
            "e.g. rerun+http://127.0.0.1:9876/proxy"
        ),
    )
    parser.add_argument(
        "--rerun-save",
        type=Path,
        help="Write a Rerun .rrd recording of live inference telemetry",
    )
    parser.add_argument(
        "--rerun-camera-fps",
        type=float,
        default=10.0,
        help="Maximum camera image logging rate for Rerun",
    )
    parser.add_argument(
        "--rerun-max-queue",
        type=int,
        default=512,
        help="Maximum queued Rerun telemetry events before dropping new events",
    )
    parser.add_argument("--arm-key", default="space")
    parser.add_argument("--reset-key", default="r", help="Reset policy state and clear queued actions without exiting")
    parser.add_argument("--home-key", default="h", help="Disarm policy and move followers to configured home positions")
    parser.add_argument("--emergency-stop-key", default="q")
    parser.add_argument("--dry-run", action="store_true", help="Run policy and print states without sending robot commands")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--check-cameras", action="store_true", help="Validate configured cameras and exit")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_inference(
    argv: list[str] | None = None,
    *,
    checkpoint_required: bool = True,
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
    runtime_adapter_builder: Callable[[argparse.Namespace, str], tuple[Any, Any]] | None = None,
    policy_label: str = "LeRobot",
) -> None:
    parser = build_arg_parser(checkpoint_required=checkpoint_required)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    config = load_config(args.config)

    required = ["left_robot_port", "right_robot_port"]
    missing = [name for name in required if cfg_value(args, config, name) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    task_description = cfg_value(args, config, "task_description")
    if not task_description:
        parser.error("--task-description or task_description in config is required")
    policy_action_hz = float(cfg_value(args, config, "policy_action_hz", DEFAULT_POLICY_ACTION_HZ))
    action_bridge_duration_s = float(
        cfg_value(args, config, "action_bridge_duration_s", DEFAULT_ACTION_BRIDGE_DURATION_S)
    )
    if args.arm_key == args.emergency_stop_key:
        parser.error("arm_key must not match emergency_stop_key")
    if args.reset_key in {args.arm_key, args.emergency_stop_key}:
        parser.error("reset_key must not match arm_key or emergency_stop_key")
    if args.home_key in {args.arm_key, args.reset_key, args.emergency_stop_key}:
        parser.error("home_key must not match arm_key, reset_key, or emergency_stop_key")
    capture_dataset = bool(getattr(args, "capture_dataset", False))
    failure_output_dir = getattr(args, "failure_output_dir", None)
    success_output_dir = getattr(args, "success_output_dir", None)
    failure_key = getattr(args, "failure_key", None)
    success_key = getattr(args, "success_key", None)
    if capture_dataset and (failure_output_dir is None or success_output_dir is None):
        parser.error("--capture-dataset requires --failure-output-dir and --success-output-dir")
    if not capture_dataset and (failure_output_dir is not None or success_output_dir is not None):
        parser.error("--failure-output-dir and --success-output-dir require --capture-dataset")
    if capture_dataset:
        deployment_keys = {args.arm_key, args.reset_key, args.home_key, args.emergency_stop_key}
        if failure_key == success_key:
            parser.error("failure_key must not match success_key")
        if failure_key in deployment_keys or success_key in deployment_keys:
            parser.error("failure_key and success_key must not match deployment control keys")
    if args.robot_fps is not None and args.robot_fps <= 0:
        parser.error("robot_fps must be > 0")
    if not np.isfinite(policy_action_hz) or policy_action_hz <= 0:
        parser.error("policy_action_hz must be > 0")
    if not np.isfinite(action_bridge_duration_s) or action_bridge_duration_s < 0:
        parser.error("action_bridge_duration_s must be >= 0")
    if not args.disable_action_age_check and args.max_action_age_s <= 0:
        parser.error("max_action_age_s must be > 0")
    if args.execution_horizon < 1:
        parser.error("execution_horizon must be >= 1")
    if args.replan_threshold < 0:
        parser.error("replan_threshold must be >= 0")
    if args.rerun_camera_fps <= 0:
        parser.error("rerun_camera_fps must be > 0")
    if args.rerun_max_queue < 1:
        parser.error("rerun_max_queue must be >= 1")

    robot_fps = int(cfg_value(args, config, "robot_fps", 60))
    camera_cfgs = build_camera_configs(args, config)
    try:
        cameras = MultiCameraManager(camera_cfgs)
    except CameraConfigError as exc:
        parser.error(str(exc))
    if args.check_cameras:
        raise SystemExit(0 if print_camera_preflight(cameras, open_cameras=True) else 1)

    robot_cfg = BimanualRobotConfig(
        left_port=cfg_value(args, config, "left_robot_port"),
        right_port=cfg_value(args, config, "right_robot_port"),
        left_id=calibration_id(config, "left_follower", "left_follower"),
        right_id=calibration_id(config, "right_follower", "right_follower"),
        calibration_dir=Path(cfg_value(args, config, "calibration_dir")).expanduser()
        if cfg_value(args, config, "calibration_dir")
        else None,
        communication_timeout_s=float(cfg_value(args, config, "communication_timeout_s", 0.25)),
        joint_limits=parse_joint_limits(config.get("joint_limits")),
    )

    if runtime_adapter_builder is None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required")
        runtime = PolicyRuntime(args.checkpoint.expanduser(), device=args.device)
        runtime.load()
        adapter = ObservationAdapter(task_description=str(task_description), image_feature_keys=runtime.image_feature_keys)
    else:
        runtime, adapter = runtime_adapter_builder(args, str(task_description))
        runtime.load()
    max_action_age_s = None if args.disable_action_age_check else float(args.max_action_age_s)
    rollout_capture = None
    if capture_dataset:
        assert failure_output_dir is not None and success_output_dir is not None
        failure_output_dir = failure_output_dir.expanduser()
        success_output_dir = success_output_dir.expanduser()
        staging_dir = success_output_dir.parent / ".maniflow-rollout-staging"
        for path in (failure_output_dir, success_output_dir, staging_dir):
            path.mkdir(parents=True, exist_ok=True)
        devices = {os.stat(path).st_dev for path in (failure_output_dir, success_output_dir, staging_dir)}
        if len(devices) != 1:
            parser.error("success, failure, and rollout staging directories must be on the same filesystem")
        checkpoint = getattr(runtime, "server_metadata", {}).get("checkpoint")
        camera_fps = int(cfg_value(args, config, "camera_fps", 30))
        rollout_capture = AutonomousRolloutCapture(
            recorder=EpisodeRecorder(
                RecorderConfig(
                    output_dir=staging_dir,
                    task_description=str(task_description),
                    camera_fps=camera_fps,
                    dataset_metadata={
                        "episode_type": "autonomous_rollout",
                        "policy_checkpoint": checkpoint,
                        "robot_fps": robot_fps,
                        "policy_action_hz": policy_action_hz,
                        "capture_hz": policy_action_hz,
                        "action_bridge_duration_s": action_bridge_duration_s,
                    },
                )
            ),
            success_output_dir=success_output_dir,
            failure_output_dir=failure_output_dir,
            policy_checkpoint=checkpoint,
            capture_hz=policy_action_hz,
        )
    debug_trace = None
    if args.debug_trace_dir is not None:
        debug_trace = DebugTraceWriter(
            args.debug_trace_dir,
            metadata={
                "policy_runtime": policy_label,
                "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                "task_description": str(task_description),
                "robot_fps": robot_fps,
                "policy_action_hz": policy_action_hz,
                "action_bridge_duration_s": action_bridge_duration_s,
                "image_feature_keys": runtime.image_feature_keys,
                "joint_names": list(DEFAULT_JOINT_NAMES),
                "max_action_age_s": max_action_age_s,
                "disable_action_age_check": bool(args.disable_action_age_check),
                "max_action_delta": args.max_action_delta,
                "execution_horizon": int(args.execution_horizon),
                "replan_threshold": int(args.replan_threshold),
                "chunk_execution_mode": args.chunk_execution_mode,
                "reset_key": args.reset_key,
                "home_key": args.home_key,
                "dry_run": bool(args.dry_run),
                "device": args.device,
            },
        )
    live_visualizer = None
    if args.rerun_live or args.rerun_connect_grpc is not None or args.rerun_save is not None:
        live_visualizer = RerunLiveVisualizer(
            RerunTelemetryConfig(
                spawn=bool(args.rerun_live),
                connect_grpc_url=args.rerun_connect_grpc,
                save_path=args.rerun_save.expanduser() if args.rerun_save is not None else None,
                camera_fps=float(args.rerun_camera_fps),
                max_queue=int(args.rerun_max_queue),
                policy_action_hz=policy_action_hz,
            ),
            joint_names=DEFAULT_JOINT_NAMES,
        )
        live_visualizer.start()

    robot = BimanualRobot.from_lerobot(robot_cfg)
    observation_buffer = LatestObservationBuffer()
    action_buffer = ActionChunkBuffer(
        policy_action_hz=policy_action_hz,
        bridge_duration_s=action_bridge_duration_s,
    )
    armed = threading.Event()
    reset_requested = threading.Event()
    policy_stop_requested = threading.Event()
    worker = PolicyWorker(
        runtime=runtime,
        observation_adapter=adapter,
        observation_buffer=observation_buffer,
        action_buffer=action_buffer,
        armed=armed,
        reset_requested=reset_requested,
        stop_requested=policy_stop_requested,
        debug_trace=debug_trace,
        live_visualizer=live_visualizer,
        execution_horizon=int(args.execution_horizon),
        replan_threshold=int(args.replan_threshold),
        chunk_execution_mode=args.chunk_execution_mode,
        wait_timeout_s=float(args.policy_wait_timeout_s),
    )
    flag = StopFlag()
    hotkeys = DeploymentHotkeys()
    install_signal_handlers(flag)
    listener = install_keyboard_controls(
        flag,
        hotkeys,
        arm_key=args.arm_key,
        reset_key=args.reset_key,
        home_key=args.home_key,
        emergency_stop_key=args.emergency_stop_key,
        failure_key=failure_key if capture_dataset else None,
        success_key=success_key if capture_dataset else None,
    )

    print(f"Policy runtime: {policy_label}")
    if args.checkpoint is not None:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {task_description}")
    print(f"Chunk execution mode: {args.chunk_execution_mode}")
    print(f"Policy trajectory: {policy_action_hz:.2f} Hz knots, {robot_fps} Hz robot loop")
    print(f"Action 0 bridge: {action_bridge_duration_s:.3f} s")
    print(
        f"Controls: '{args.arm_key}' arm/disarm, '{args.reset_key}' reset policy, "
        f"'{args.home_key}' disarm+home, '{args.emergency_stop_key}' emergency stop"
    )
    if rollout_capture is not None:
        print(
            f"Rollout capture: '{failure_key}' failure -> {failure_output_dir}, "
            f"'{success_key}' success -> {success_output_dir}"
        )
    if args.dry_run:
        print("Dry run: policy commands will not be sent to the robot")
    if debug_trace is not None:
        print(f"Debug trace: {debug_trace.run_dir}")
    if live_visualizer is not None:
        destinations = []
        if args.rerun_live:
            destinations.append("viewer")
        if args.rerun_connect_grpc is not None:
            destinations.append(args.rerun_connect_grpc)
        if args.rerun_save is not None:
            destinations.append(str(args.rerun_save))
        print(f"Rerun live telemetry: {', '.join(destinations)}")

    try:
        cameras.start()
        robot.connect(calibrate=args.calibrate)
        move_followers_home(robot, config, args)
        runtime.reset()
        worker.start()
        run_control_loop(
            robot=robot,
            cameras=cameras,
            policy_worker=worker,
            observation_buffer=observation_buffer,
            action_buffer=action_buffer,
            armed=armed,
            reset_requested=reset_requested,
            policy_stop_requested=policy_stop_requested,
            hotkeys=hotkeys,
            robot_fps=robot_fps,
            image_feature_keys=runtime.image_feature_keys,
            max_action_age_s=max_action_age_s,
            max_action_delta=args.max_action_delta,
            home_callback=lambda: move_followers_home_now(robot, config, args),
            dry_run=bool(args.dry_run),
            flag=flag,
            debug_trace=debug_trace,
            live_visualizer=live_visualizer,
            rollout_capture=rollout_capture,
        )
    finally:
        logger.info("Shutting down inference: %s", flag.reason or "normal exit")
        policy_stop_requested.set()
        worker.stop()
        robot.disconnect()
        cameras.stop()
        if listener is not None:
            listener.stop()
        if live_visualizer is not None:
            live_visualizer.close()
            if live_visualizer.dropped_events:
                logger.warning("Dropped %d Rerun telemetry events", live_visualizer.dropped_events)
        if debug_trace is not None:
            debug_trace.close()
        if rollout_capture is not None:
            rollout_capture.discard()


def main(argv: list[str] | None = None) -> None:
    run_inference(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
