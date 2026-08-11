"""Corrections-only HIL collection while deploying a remote ManiFlow policy."""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np

from bimanual_collection.bimanual_teleop import (
    build_camera_configs,
    calibration_id,
    cfg_value,
    default_home_position_file,
    load_config,
    move_followers_home,
    parse_joint_limits,
    print_camera_preflight,
    setup_logging,
    teleop_value,
)
from bimanual_collection.control.clutch import BimanualClutchController
from bimanual_collection.hardware.bimanual_leader import BimanualLeader, BimanualLeaderConfig
from bimanual_collection.hardware.bimanual_robot import DEFAULT_JOINT_NAMES, BimanualRobot, BimanualRobotConfig
from bimanual_collection.hardware.cameras import CameraConfigError, MultiCameraManager
from bimanual_collection.inference import (
    ActionChunkBuffer,
    CHUNK_EXECUTION_MODES,
    DEFAULT_ACTION_BRIDGE_DURATION_S,
    DEFAULT_POLICY_ACTION_HZ,
    ChunkExecutionMode,
    LatestObservationBuffer,
    ObservationSnapshot,
    PeriodicCaptureGate,
    PolicyWorker,
    StopFlag,
    bimanual_joint_vector,
    camera_matches_valid,
    install_signal_handlers,
    precise_sleep_until,
    split_bimanual_action,
    validate_action_delta,
)
from bimanual_collection.maniflow_inference import _normalize_server_url
from bimanual_collection.maniflow_remote import (
    DEFAULT_MANIFLOW_CAMERAS,
    ManiFlowRemoteObservationAdapter,
    ManiFlowRemotePolicyRuntime,
)
from bimanual_collection.recording.episode import TimestepSample, gripper_state
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig


logger = logging.getLogger(__name__)


class HILPhase:
    READY_TO_ARM = "READY_TO_ARM"
    AUTONOMOUS = "AUTONOMOUS"
    PAUSED = "PAUSED"
    PRE_CORRECTION = "PRE_CORRECTION"
    CORRECTING = "CORRECTING"
    ROLLOUT_TERMINATED = "ROLLOUT_TERMINATED"
    ESTOP = "ESTOP"


PedalBackend = Literal["keyboard", "evdev"]
HILProtocol = Literal["continuous", "rac", "bounded"]
HIL_PROTOCOLS = ("continuous", "rac", "bounded")


@dataclass(frozen=True)
class HILControlConfig:
    backend: PedalBackend = "evdev"
    device: str | None = None
    debounce_s: float = 0.05
    pause_resume: str = "KEY_1"
    correction: str = "KEY_2"
    clutch: str = "KEY_3"
    arm: str = "space"
    reset: str = "r"


@dataclass(frozen=True)
class HILProtocolConfig:
    protocol: HILProtocol = "continuous"
    max_interventions_per_rollout: int = 2


@dataclass
class HILRolloutState:
    rollout_id: str
    intervention_count: int = 0

    @classmethod
    def fresh(cls) -> "HILRolloutState":
        return cls(rollout_id=str(uuid.uuid4()))


class HILGenerationGate:
    """Thread-safe policy generation gate shared with the inference worker."""

    def __init__(self, phase: str = HILPhase.AUTONOMOUS) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._phase = phase

    def current_generation(self) -> int:
        with self._lock:
            return self._generation

    def bump(self, *, phase: str | None = None) -> int:
        with self._lock:
            self._generation += 1
            if phase is not None:
                self._phase = phase
            return self._generation

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def accepts_response(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation and self._phase == HILPhase.AUTONOMOUS


@dataclass(frozen=True)
class HILControlSnapshot:
    pause_resume_edge: bool
    correction_edge: bool
    arm_edge: bool
    reset_edge: bool
    clutch_active: bool
    failed: bool
    error: str | None


class HILControls:
    """Three-button HIL input: pause/resume, correction toggle, clutch hold."""

    def __init__(self, config: HILControlConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._pause_resume_edge = False
        self._correction_edge = False
        self._arm_edge = False
        self._reset_edge = False
        self._clutch_active = False
        self._pause_pressed = False
        self._correction_pressed = False
        self._arm_pressed = False
        self._reset_pressed = False
        self._failed = False
        self._error: str | None = None
        self._last_change: dict[str, float] = {
            "pause_resume": 0.0,
            "correction": 0.0,
            "arm": 0.0,
            "reset": 0.0,
            "clutch": 0.0,
        }
        self._listener: Any | None = None
        self._utility_listener: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self.config.backend == "keyboard":
            self._start_keyboard()
        elif self.config.backend == "evdev":
            self._start_evdev()
            self._start_keyboard_utilities()
        else:
            raise ValueError(f"Unsupported HIL pedal backend: {self.config.backend}")

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
            self._listener = None
        if self._utility_listener is not None:
            with contextlib.suppress(Exception):
                self._utility_listener.stop()
            self._utility_listener = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> HILControlSnapshot:
        with self._lock:
            snapshot = HILControlSnapshot(
                pause_resume_edge=self._pause_resume_edge,
                correction_edge=self._correction_edge,
                arm_edge=self._arm_edge,
                reset_edge=self._reset_edge,
                clutch_active=self._clutch_active,
                failed=self._failed,
                error=self._error,
            )
            self._pause_resume_edge = False
            self._correction_edge = False
            self._arm_edge = False
            self._reset_edge = False
            return snapshot

    def _mark_failed(self, message: str) -> None:
        with self._lock:
            self._failed = True
            self._error = message
        logger.error("HIL pedal input failed: %s", message)
        print(f"HIL pedal input failed: {message}")

    def _debounced(self, control: str) -> bool:
        now_s = time.monotonic()
        if now_s - self._last_change[control] < self.config.debounce_s:
            return False
        self._last_change[control] = now_s
        return True

    def _request_edge(self, control: Literal["pause_resume", "correction", "arm", "reset"]) -> None:
        if not self._debounced(control):
            return
        with self._lock:
            if control == "pause_resume":
                self._pause_resume_edge = True
            elif control == "correction":
                self._correction_edge = True
            elif control == "arm":
                self._arm_edge = True
            else:
                self._reset_edge = True

    def _set_clutch(self, active: bool) -> None:
        if active and not self._debounced("clutch"):
            return
        with self._lock:
            changed = self._clutch_active != active
            self._clutch_active = active
            if not active:
                self._last_change["clutch"] = time.monotonic()
        if changed:
            print(f"HIL clutch {'engaged' if active else 'released'}")

    @staticmethod
    def _normalize_keyboard_key(key: Any) -> str:
        try:
            char = str(key.char)
            return "space" if char == " " else char
        except AttributeError:
            return str(key).replace("Key.", "")

    def _start_keyboard(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            self._mark_failed(f"pynput unavailable: {exc}")
            return

        def on_press(key: Any) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.pause_resume:
                with self._lock:
                    if self._pause_pressed:
                        return
                    self._pause_pressed = True
                self._request_edge("pause_resume")
            elif name == self.config.correction:
                with self._lock:
                    if self._correction_pressed:
                        return
                    self._correction_pressed = True
                self._request_edge("correction")
            elif name == self.config.arm:
                with self._lock:
                    if self._arm_pressed:
                        return
                    self._arm_pressed = True
                self._request_edge("arm")
            elif name == self.config.reset:
                with self._lock:
                    if self._reset_pressed:
                        return
                    self._reset_pressed = True
                self._request_edge("reset")
            elif name == self.config.clutch:
                self._set_clutch(True)

        def on_release(key: Any) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.pause_resume:
                with self._lock:
                    self._pause_pressed = False
            elif name == self.config.correction:
                with self._lock:
                    self._correction_pressed = False
            elif name == self.config.arm:
                with self._lock:
                    self._arm_pressed = False
            elif name == self.config.reset:
                with self._lock:
                    self._reset_pressed = False
            elif name == self.config.clutch:
                self._set_clutch(False)

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
        except Exception as exc:
            self._mark_failed(f"keyboard listener failed: {exc}")

    def _start_keyboard_utilities(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            logger.warning("HIL arm/reset keyboard utilities unavailable: %s", exc)
            return

        def on_press(key: Any) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.arm:
                with self._lock:
                    if self._arm_pressed:
                        return
                    self._arm_pressed = True
                self._request_edge("arm")
            elif name == self.config.reset:
                with self._lock:
                    if self._reset_pressed:
                        return
                    self._reset_pressed = True
                self._request_edge("reset")

        def on_release(key: Any) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.arm:
                with self._lock:
                    self._arm_pressed = False
            elif name == self.config.reset:
                with self._lock:
                    self._reset_pressed = False

        try:
            self._utility_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._utility_listener.start()
        except Exception as exc:
            logger.warning("HIL arm/reset keyboard listener failed: %s", exc)

    def _start_evdev(self) -> None:
        if not self.config.device:
            self._mark_failed("evdev backend requires --hil-pedal-device")
            return
        self._thread = threading.Thread(target=self._evdev_loop, name="hil-pedal-evdev", daemon=True)
        self._thread.start()

    def _evdev_loop(self) -> None:
        try:
            from evdev import InputDevice, categorize, ecodes  # type: ignore[import-not-found]
        except Exception as exc:
            self._mark_failed(f"evdev unavailable: {exc}")
            return

        try:
            assert self.config.device is not None

            def resolve_key_code(value: str) -> int:
                key_name = str(value)
                if key_name in ecodes.ecodes:
                    return int(ecodes.ecodes[key_name])
                if key_name.startswith("KEY") and not key_name.startswith("KEY_"):
                    normalized = f"KEY_{key_name.removeprefix('KEY')}"
                    if normalized in ecodes.ecodes:
                        return int(ecodes.ecodes[normalized])
                return int(key_name)

            device = InputDevice(str(Path(self.config.device).expanduser()))
            pause_code = resolve_key_code(self.config.pause_resume)
            correction_code = resolve_key_code(self.config.correction)
            clutch_code = resolve_key_code(self.config.clutch)
            for event in device.read_loop():
                if self._stop.is_set():
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = categorize(event)
                code = key_event.scancode
                if code == pause_code and key_event.keystate == key_event.key_down:
                    self._request_edge("pause_resume")
                elif code == correction_code and key_event.keystate == key_event.key_down:
                    self._request_edge("correction")
                elif code == clutch_code:
                    if key_event.keystate == key_event.key_down:
                        self._set_clutch(True)
                    elif key_event.keystate == key_event.key_up:
                        self._set_clutch(False)
        except Exception as exc:
            if not self._stop.is_set():
                self._mark_failed(str(exc))


def next_hil_phase(
    phase: str,
    snapshot: HILControlSnapshot,
    *,
    protocol: HILProtocol = "continuous",
    intervention_count: int = 0,
    max_interventions_per_rollout: int = 2,
) -> str:
    if snapshot.reset_edge:
        return HILPhase.READY_TO_ARM
    if snapshot.arm_edge:
        if phase == HILPhase.READY_TO_ARM:
            return HILPhase.AUTONOMOUS
        return phase
    if snapshot.correction_edge:
        if phase == HILPhase.PAUSED:
            return HILPhase.PRE_CORRECTION
        if phase == HILPhase.PRE_CORRECTION:
            return HILPhase.CORRECTING
        if phase == HILPhase.CORRECTING:
            return next_phase_after_saved_correction(
                HILProtocolConfig(protocol=protocol, max_interventions_per_rollout=max_interventions_per_rollout),
                intervention_count + 1,
            )
    if snapshot.pause_resume_edge:
        if phase == HILPhase.AUTONOMOUS:
            return HILPhase.PAUSED
        if phase == HILPhase.PAUSED:
            return HILPhase.AUTONOMOUS
        if phase == HILPhase.PRE_CORRECTION:
            return HILPhase.PAUSED
    return phase


def validate_hil_protocol_config(config: HILProtocolConfig) -> None:
    if config.protocol not in HIL_PROTOCOLS:
        raise ValueError(f"hil_protocol must be one of {HIL_PROTOCOLS}, got {config.protocol!r}")
    if config.max_interventions_per_rollout < 1:
        raise ValueError("hil_max_interventions_per_rollout must be >= 1")


def next_phase_after_saved_correction(config: HILProtocolConfig, rollout_intervention_count: int) -> str:
    validate_hil_protocol_config(config)
    if config.protocol == "continuous":
        return HILPhase.PAUSED
    if config.protocol == "rac":
        return HILPhase.ROLLOUT_TERMINATED
    if rollout_intervention_count >= config.max_interventions_per_rollout:
        return HILPhase.ROLLOUT_TERMINATED
    return HILPhase.PAUSED


def correction_terminates_rollout(config: HILProtocolConfig, next_rollout_intervention_count: int) -> bool:
    return next_phase_after_saved_correction(config, next_rollout_intervention_count) == HILPhase.ROLLOUT_TERMINATED


def correction_episode_metadata(
    *,
    protocol_config: HILProtocolConfig,
    rollout: HILRolloutState,
    policy_server: str,
    policy_generation_id: int,
) -> dict[str, Any]:
    intervention_index = rollout.intervention_count
    next_count = intervention_index + 1
    metadata: dict[str, Any] = {
        "collection_type": "hil",
        "hil_protocol": protocol_config.protocol,
        "contains_autonomous_actions": False,
        "intervention_index": intervention_index,
        "rollout_intervention_count": next_count,
        "prior_human_intervention": intervention_index > 0,
        "rollout_terminated_after_intervention": correction_terminates_rollout(protocol_config, next_count),
        "policy_name": "maniflow",
        "policy_server": policy_server,
        "rollout_id": rollout.rollout_id,
        "policy_generation_id": policy_generation_id,
        "failure_reason": None,
        "subtask": None,
        "correction_success": None,
    }
    if protocol_config.protocol == "bounded":
        metadata["max_interventions_per_rollout"] = protocol_config.max_interventions_per_rollout
    return metadata


def build_hil_protocol_config(args: argparse.Namespace, config: dict[str, Any]) -> HILProtocolConfig:
    raw = config.get("maniflow_hil", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("maniflow_hil config must be a mapping")
    protocol = str(args.hil_protocol or raw.get("hil_protocol") or raw.get("protocol", "continuous"))
    max_interventions = int(
        args.hil_max_interventions_per_rollout
        if args.hil_max_interventions_per_rollout is not None
        else raw.get("hil_max_interventions_per_rollout", raw.get("max_interventions_per_rollout", 2))
    )
    protocol_config = HILProtocolConfig(
        protocol=cast(HILProtocol, protocol),
        max_interventions_per_rollout=max_interventions,
    )
    validate_hil_protocol_config(protocol_config)
    return protocol_config


def build_hil_control_config(args: argparse.Namespace, config: dict[str, Any]) -> HILControlConfig:
    raw = config.get("maniflow_hil", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("maniflow_hil config must be a mapping")
    backend = str(args.hil_pedal_backend or raw.get("pedal_backend", "evdev"))
    if backend not in ("keyboard", "evdev"):
        raise ValueError(f"maniflow_hil.pedal_backend must be 'keyboard' or 'evdev', got {backend!r}")
    return HILControlConfig(
        backend=cast(PedalBackend, backend),
        device=args.hil_pedal_device or raw.get("pedal_device"),
        debounce_s=float(
            args.hil_pedal_debounce_s
            if args.hil_pedal_debounce_s is not None
            else raw.get("pedal_debounce_s", 0.05)
        ),
        pause_resume=str(args.hil_pause_code or raw.get("pause_code", "KEY_1")),
        correction=str(args.hil_correction_code or raw.get("correction_code", "KEY_2")),
        clutch=str(args.hil_clutch_code or raw.get("clutch_code", "KEY_3")),
        arm=str(args.hil_arm_code or raw.get("arm_code", "space")),
        reset=str(args.hil_reset_code or raw.get("reset_code", "r")),
    )


def _event(recorder: EpisodeRecorder | None, episode_id: str, name: str, **payload: Any) -> None:
    if recorder is None:
        return
    recorder.add_event(
        {
            "episode_id": episode_id,
            "monotonic_timestamp_s": time.monotonic(),
            "wall_timestamp_s": time.time(),
            "event": name,
            **payload,
        }
    )


def _start_correction(
    recorder: EpisodeRecorder,
    episode_id: str | None,
    *,
    metadata: dict[str, Any],
) -> str:
    if recorder.is_recording:
        return episode_id or recorder.episode_id or "hil-correction"
    new_episode_id = recorder.start(extra=metadata)
    print(f"Started HIL correction {recorder.episode_label} -> {recorder.current_final_dir}")
    _event(recorder, new_episode_id, "hil_correction_started", **metadata)
    return new_episode_id


def _finish_correction(recorder: EpisodeRecorder, reason: str) -> tuple[bool, int]:
    if not recorder.is_recording:
        return False, 0
    episode_label = recorder.episode_label
    samples = recorder.sample_count
    if samples <= 0:
        recorder.discard()
        print(f"Discarded empty HIL correction {episode_label}; intervention was not counted")
        return False, 0
    path = recorder.stop_and_save(success=True, operator_notes=reason)
    print(f"Saved HIL correction {episode_label}: {samples} samples -> {path}")
    return True, samples


def _save_correction(recorder: EpisodeRecorder, reason: str) -> None:
    _finish_correction(recorder, reason)


def _discard_active_correction(recorder: EpisodeRecorder, reason: str) -> bool:
    if not recorder.is_recording:
        return False
    label = recorder.episode_label
    recorder.discard()
    print(f"Discarded HIL correction {label}: {reason}")
    return True


def install_keyboard_estop(flag: StopFlag, key_name: str) -> Any | None:
    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Keyboard estop unavailable: %s", exc)
        return None

    def normalize(key: Any) -> str:
        try:
            return str(key.char)
        except AttributeError:
            return str(key).replace("Key.", "")

    def on_press(key: Any) -> None:
        if normalize(key) == key_name:
            flag.request(f"emergency stop key '{key_name}'")

    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    except Exception as exc:
        logger.warning("Keyboard estop listener failed: %s", exc)
        return None


def run_hil_loop(
    *,
    robot: BimanualRobot,
    leader: BimanualLeader,
    cameras: MultiCameraManager,
    controls: HILControls,
    recorder: EpisodeRecorder,
    policy_worker: PolicyWorker,
    protocol_config: HILProtocolConfig,
    generation_gate: HILGenerationGate,
    policy_server: str,
    home_followers: Callable[[], None],
    observation_buffer: LatestObservationBuffer,
    action_buffer: ActionChunkBuffer,
    armed: threading.Event,
    reset_requested: threading.Event,
    policy_stop_requested: threading.Event,
    image_feature_keys: list[str],
    robot_fps: int,
    capture_hz: float,
    max_action_age_s: float | None,
    max_action_delta: float | None,
    dry_run: bool,
    flag: StopFlag,
) -> None:
    period_s = 1.0 / robot_fps
    phase = HILPhase.READY_TO_ARM
    generation_gate.set_phase(phase)
    controller = BimanualClutchController(startup_alignment="hold_current")
    rollout = HILRolloutState.fresh()
    armed.clear()
    correction_episode_id: str | None = None
    correction_metadata: dict[str, Any] = {}
    sample_index = 0
    correction_capture_gate = PeriodicCaptureGate(capture_hz)
    previous_loop_start = time.monotonic()
    print(
        "HIL phase -> READY_TO_ARM "
        f"(protocol={protocol_config.protocol}, rollout={rollout.rollout_id}, interventions=0). "
        f"Press {controls.config.arm} to start autonomy."
    )

    while not flag.stop and not policy_stop_requested.is_set():
        loop_start = time.monotonic()
        snapshot = controls.snapshot()
        if snapshot.failed:
            flag.request(f"HIL pedal failure: {snapshot.error}")
            break

        if snapshot.reset_edge:
            old_phase = phase
            armed.clear()
            _discard_active_correction(recorder, "reset/cancel requested")
            correction_episode_id = None
            correction_metadata = {}
            sample_index = 0
            action_buffer.clear()
            observation_buffer.clear()
            reset_requested.set()
            rollout = HILRolloutState.fresh()
            controller = BimanualClutchController(startup_alignment="hold_current")
            phase = HILPhase.READY_TO_ARM
            generation_id = generation_gate.bump(phase=phase)
            print(
                f"HIL reset requested by {controls.config.reset}. "
                "Cancelling active correction if any, clearing policy state, and homing followers."
            )
            try:
                if not dry_run:
                    home_followers()
            except Exception as exc:
                flag.request(f"reset/home failure: {exc}")
                logger.exception("HIL reset/home failed; stopping")
                break
            action_buffer.clear()
            observation_buffer.clear()
            print(
                f"HIL phase -> {phase} "
                f"(rollout={rollout.rollout_id}, generation={generation_id}). "
                f"Physically reset the scene, then press {controls.config.arm} to start the next rollout."
            )
            logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            precise_sleep_until(loop_start + period_s)
            previous_loop_start = loop_start
            continue

        if snapshot.arm_edge:
            if phase == HILPhase.READY_TO_ARM:
                old_phase = phase
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                phase = HILPhase.AUTONOMOUS
                generation_id = generation_gate.bump(phase=phase)
                armed.set()
                print(
                    f"HIL phase -> {phase} "
                    f"(protocol={protocol_config.protocol}, rollout={rollout.rollout_id}, "
                    f"generation={generation_id})"
                )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            else:
                print(f"Ignoring {controls.config.arm}: only valid from {HILPhase.READY_TO_ARM}")

        if snapshot.correction_edge:
            if phase == HILPhase.PAUSED:
                old_phase = phase
                phase = HILPhase.PRE_CORRECTION
                armed.clear()
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                generation_id = generation_gate.bump(phase=phase)
                controller = BimanualClutchController(startup_alignment="hold_current")
                sample_index = 0
                print(
                    f"HIL phase -> {phase} "
                    f"(leader setup, not recording, generation={generation_id}). "
                    f"Press {controls.config.correction} again to start recording."
                )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.PRE_CORRECTION:
                old_phase = phase
                phase = HILPhase.CORRECTING
                armed.clear()
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                generation_id = generation_gate.bump(phase=phase)
                correction_metadata = {
                    **correction_episode_metadata(
                        protocol_config=protocol_config,
                        rollout=rollout,
                        policy_server=policy_server,
                        policy_generation_id=generation_id,
                    ),
                    "capture_hz": capture_hz,
                }
                correction_episode_id = _start_correction(
                    recorder,
                    correction_episode_id,
                    metadata=correction_metadata,
                )
                sample_index = 0
                correction_capture_gate.reset()
                print(
                    f"HIL phase -> {phase} "
                    f"(intervention={rollout.intervention_count + 1}, generation={generation_id})"
                )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.CORRECTING:
                old_phase = phase
                _event(recorder, correction_episode_id or "hil-correction", "hil_correction_stopped")
                saved = False
                try:
                    saved, _samples = _finish_correction(recorder, reason="Saved by HIL correction control")
                except Exception as exc:
                    logger.exception("Failed to save HIL correction")
                    print(f"Failed to save HIL correction: {exc}")
                    if recorder.is_recording:
                        with contextlib.suppress(Exception):
                            recorder.discard()
                correction_episode_id = None
                correction_metadata = {}
                sample_index = 0
                if saved:
                    rollout.intervention_count += 1
                    phase = next_phase_after_saved_correction(protocol_config, rollout.intervention_count)
                else:
                    phase = HILPhase.PAUSED
                armed.clear()
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                generation_id = generation_gate.bump(phase=phase)
                print(
                    f"HIL phase -> {phase} "
                    f"(protocol={protocol_config.protocol}, interventions={rollout.intervention_count}, "
                    f"generation={generation_id})"
                )
                if phase == HILPhase.ROLLOUT_TERMINATED:
                    saved_message = (
                        "RaC correction saved. Rollout terminated."
                        if protocol_config.protocol == "rac"
                        else "Correction limit reached. Rollout terminated."
                    )
                    print(
                        f"{saved_message} "
                        f"Press {controls.config.reset} to home/reset, then {controls.config.arm} to begin a new rollout."
                    )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.AUTONOMOUS:
                print(
                    f"Ignoring {controls.config.correction}: pause with {controls.config.pause_resume} "
                    "before correction setup"
                )
            elif phase == HILPhase.READY_TO_ARM:
                print(f"Ignoring {controls.config.correction}: press {controls.config.arm} to start autonomy first")
            elif phase == HILPhase.ROLLOUT_TERMINATED:
                print(f"Ignoring {controls.config.correction}: rollout is terminated; press {controls.config.reset} to reset/home")
        elif snapshot.pause_resume_edge:
            if phase == HILPhase.AUTONOMOUS:
                old_phase = phase
                phase = HILPhase.PAUSED
                armed.clear()
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                generation_id = generation_gate.bump(phase=phase)
                print(f"HIL phase -> {phase} (generation={generation_id})")
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.PAUSED:
                old_phase = phase
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                phase = HILPhase.AUTONOMOUS
                generation_id = generation_gate.bump(phase=phase)
                armed.set()
                print(
                    f"HIL phase -> {phase} "
                    f"(protocol={protocol_config.protocol}, interventions={rollout.intervention_count}, "
                    f"generation={generation_id})"
                )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.PRE_CORRECTION:
                old_phase = phase
                phase = HILPhase.PAUSED
                action_buffer.clear()
                observation_buffer.clear()
                reset_requested.set()
                generation_id = generation_gate.bump(phase=phase)
                print(
                    f"HIL phase -> {phase} "
                    f"(pre-correction setup cancelled, generation={generation_id})"
                )
                logger.info("HIL phase transition: %s -> %s", old_phase, phase)
            elif phase == HILPhase.CORRECTING:
                print(f"Ignoring {controls.config.pause_resume}: finish correction with {controls.config.correction} first")
            elif phase == HILPhase.ROLLOUT_TERMINATED:
                print(f"Ignoring {controls.config.pause_resume}: press {controls.config.reset} to reset/home first")
            elif phase == HILPhase.READY_TO_ARM:
                print(f"Ignoring {controls.config.pause_resume}: press {controls.config.arm} to start autonomy")

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
            logger.exception("Hardware read failed; stopping HIL")
            break

        if phase == HILPhase.AUTONOMOUS:
            if camera_matches_valid(camera_sample.matches, image_feature_keys):
                observation_buffer.publish(
                    timestamp_s=sample_timestamp,
                    follower_state=follower_state,
                    camera_matches=camera_sample.matches,
                )
            else:
                observation_buffer.clear()
                action_buffer.clear()
                reset_requested.set()

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
                if not dry_run:
                    with contextlib.suppress(Exception):
                        robot.hold_position()
                precise_sleep_until(loop_start + period_s)
                previous_loop_start = loop_start
                continue

            try:
                timed_action = popped_action.action
                left_action, right_action = split_bimanual_action(timed_action.action)
                validate_action_delta(left_action, right_action, follower_state, max_action_delta)
                if not dry_run:
                    robot.send_actions(left_action, right_action)
            except Exception as exc:
                flag.request(f"unsafe policy action: {exc}")
                logger.exception("Unsafe policy action; stopping HIL")
                break

        elif phase in (HILPhase.READY_TO_ARM, HILPhase.PAUSED):
            if not dry_run:
                with contextlib.suppress(Exception):
                    robot.hold_position()

        elif phase == HILPhase.PRE_CORRECTION:
            try:
                leader_state = leader.read()
                control = controller.update(
                    leader_state.left,
                    leader_state.right,
                    follower_state.left,
                    follower_state.right,
                    left_clutch_active=snapshot.clutch_active,
                    right_clutch_active=snapshot.clutch_active,
                    recording_paused=False,
                )
                if not dry_run:
                    robot.send_actions(control.left_action, control.right_action)
            except Exception as exc:
                flag.request(f"pre-correction setup failure: {exc}")
                logger.exception("Pre-correction setup failed; stopping HIL")
                break

        elif phase == HILPhase.ROLLOUT_TERMINATED:
            action_buffer.clear()
            if not dry_run:
                with contextlib.suppress(Exception):
                    robot.hold_position()

        elif phase == HILPhase.CORRECTING:
            try:
                leader_state = leader.read()
                control = controller.update(
                    leader_state.left,
                    leader_state.right,
                    follower_state.left,
                    follower_state.right,
                    left_clutch_active=snapshot.clutch_active,
                    right_clutch_active=snapshot.clutch_active,
                    recording_paused=False,
                )
                command = None
                if not dry_run:
                    command = robot.send_actions(control.left_action, control.right_action)
            except Exception as exc:
                flag.request(f"teleop correction failure: {exc}")
                logger.exception("Teleop correction failed; stopping HIL")
                break

            if (
                recorder.is_recording
                and control.should_record
                and camera_matches_valid(camera_sample.matches, image_feature_keys)
                and correction_capture_gate.should_capture(sample_timestamp)
            ):
                loop_duration_s = time.monotonic() - loop_start
                interval_s = loop_start - previous_loop_start
                measured_hz = 1.0 / interval_s if interval_s > 0 else 0.0
                command_left = command.left if command is not None else control.left_action
                command_right = command.right if command is not None else control.right_action
                recorder.add_sample(
                    TimestepSample(
                        episode_id=correction_episode_id or recorder.episode_id or "hil-correction",
                        timestep_index=sample_index,
                        monotonic_timestamp_s=sample_timestamp,
                        wall_timestamp_s=time.time(),
                        left_leader_joints=leader_state.left,
                        right_leader_joints=leader_state.right,
                        left_follower_joints=follower_state.left,
                        right_follower_joints=follower_state.right,
                        left_gripper_state=gripper_state(follower_state.left),
                        right_gripper_state=gripper_state(follower_state.right),
                        left_commanded_action=command_left,
                        right_commanded_action=command_right,
                        camera_matches=camera_sample.matches,
                        measured_control_hz=measured_hz,
                        loop_duration_s=loop_duration_s,
                        metadata={
                            "action_source": "hil_human_correction",
                            "hil_phase": phase,
                            "clutch_active": snapshot.clutch_active,
                            **correction_metadata,
                        },
                    )
                )
                sample_index += 1

        previous_loop_start = loop_start
        precise_sleep_until(loop_start + period_s)

    phase = HILPhase.ESTOP
    print(f"HIL phase -> {phase}: {flag.reason or 'stopped'}")
    armed.clear()
    action_buffer.clear()
    if recorder.is_recording:
        if flag.reason.startswith("hardware failure"):
            recorder.discard()
        else:
            _save_correction(recorder, reason=f"Stopped by {flag.reason or 'operator'}")
    if not dry_run and robot.is_connected:
        with contextlib.suppress(Exception):
            robot.hold_position()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--task-description")
    parser.add_argument("--left-robot-port")
    parser.add_argument("--right-robot-port")
    parser.add_argument("--left-leader-port")
    parser.add_argument("--right-leader-port")
    parser.add_argument("--overhead-camera")
    parser.add_argument("--left-wrist-camera")
    parser.add_argument("--right-wrist-camera")
    parser.add_argument("--robot-fps", type=int)
    parser.add_argument("--policy-action-hz", type=float)
    parser.add_argument("--action-bridge-duration-s", type=float)
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
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset/maniflow_hil_corrections"))
    parser.add_argument("--episode-start-number", type=int)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--maniflow-server")
    parser.add_argument("--maniflow-timeout-s", type=float)
    parser.add_argument("--maniflow-camera", action="append", dest="maniflow_cameras")
    parser.add_argument("--hil-protocol", choices=HIL_PROTOCOLS, help="HIL intervention protocol")
    parser.add_argument("--hil-max-interventions-per-rollout", type=int)
    parser.add_argument("--hil-pedal-backend", choices=["keyboard", "evdev"])
    parser.add_argument("--hil-pedal-device")
    parser.add_argument("--hil-pedal-debounce-s", type=float)
    parser.add_argument("--hil-pause-code", help="Pedal code/key for AUTONOMOUS<->PAUSED")
    parser.add_argument("--hil-correction-code", help="Pedal code/key for PAUSED<->CORRECTING")
    parser.add_argument("--hil-clutch-code", help="Held pedal code/key for both-arm clutch/recenter during correction")
    parser.add_argument("--hil-arm-code", help="Keyboard key to arm/start after homing or reset")
    parser.add_argument("--hil-reset-code", help="Keyboard key to cancel/reset/home and wait for arm")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--replan-threshold", type=int, default=5)
    parser.add_argument("--chunk-execution-mode", choices=CHUNK_EXECUTION_MODES, default=ChunkExecutionMode.FULL)
    parser.add_argument("--policy-wait-timeout-s", type=float, default=0.1)
    parser.add_argument("--max-action-age-s", type=float, default=1.0)
    parser.add_argument("--disable-action-age-check", action="store_true")
    parser.add_argument("--max-action-delta", type=float)
    parser.add_argument("--emergency-stop-key", default="q")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--check-cameras", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    config = load_config(args.config)

    required = ["left_robot_port", "right_robot_port", "left_leader_port", "right_leader_port"]
    missing = [name for name in required if cfg_value(args, config, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    task_description = cfg_value(args, config, "task_description")
    if not task_description:
        raise SystemExit("--task-description or task_description in config is required")
    policy_action_hz = float(cfg_value(args, config, "policy_action_hz", DEFAULT_POLICY_ACTION_HZ))
    action_bridge_duration_s = float(
        cfg_value(args, config, "action_bridge_duration_s", DEFAULT_ACTION_BRIDGE_DURATION_S)
    )
    if args.robot_fps is not None and args.robot_fps <= 0:
        raise SystemExit("robot_fps must be > 0")
    if not np.isfinite(policy_action_hz) or policy_action_hz <= 0:
        raise SystemExit("policy_action_hz must be > 0")
    if not np.isfinite(action_bridge_duration_s) or action_bridge_duration_s < 0:
        raise SystemExit("action_bridge_duration_s must be >= 0")
    if args.execution_horizon < 1:
        raise SystemExit("execution_horizon must be >= 1")
    if args.replan_threshold < 0:
        raise SystemExit("replan_threshold must be >= 0")
    if not args.disable_action_age_check and args.max_action_age_s <= 0:
        raise SystemExit("max_action_age_s must be > 0")
    maniflow_timeout_s = float(cfg_value(args, config, "maniflow_timeout_s", 10.0))
    if maniflow_timeout_s <= 0:
        raise SystemExit("maniflow_timeout_s must be > 0")
    hil_protocol_cfg = build_hil_protocol_config(args, config)

    robot_fps = int(cfg_value(args, config, "robot_fps", 60))
    camera_fps = int(cfg_value(args, config, "camera_fps", 30))
    camera_cfgs = build_camera_configs(args, config)
    try:
        cameras = MultiCameraManager(camera_cfgs)
    except CameraConfigError as exc:
        raise SystemExit(str(exc)) from exc
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
    leader_cfg = BimanualLeaderConfig(
        left_port=cfg_value(args, config, "left_leader_port"),
        right_port=cfg_value(args, config, "right_leader_port"),
        left_id=calibration_id(config, "left_leader", "left_leader"),
        right_id=calibration_id(config, "right_leader", "right_leader"),
        calibration_dir=Path(cfg_value(args, config, "calibration_dir")).expanduser()
        if cfg_value(args, config, "calibration_dir")
        else None,
        communication_timeout_s=float(cfg_value(args, config, "communication_timeout_s", 0.25)),
    )
    cameras_for_maniflow = tuple(
        args.maniflow_cameras or config.get("maniflow_cameras") or DEFAULT_MANIFLOW_CAMERAS
    )
    normalized_maniflow_server = _normalize_server_url(
        str(cfg_value(args, config, "maniflow_server", "http://127.0.0.1:8765"))
    )
    runtime = ManiFlowRemotePolicyRuntime(
        normalized_maniflow_server,
        timeout_s=maniflow_timeout_s,
        cameras=cameras_for_maniflow,
    )
    adapter = ManiFlowRemoteObservationAdapter(task_description=str(task_description), cameras=cameras_for_maniflow)
    runtime.load()

    hil_controls_cfg = build_hil_control_config(args, config)
    controls = HILControls(hil_controls_cfg)
    recorder = EpisodeRecorder(
        RecorderConfig(
            output_dir=args.output_dir.expanduser(),
            warmup_s=float(args.warmup_s),
            task_description=str(task_description),
            camera_fps=camera_fps,
            episode_start_number=args.episode_start_number,
            environment=config.get("environment", {}),
            robot_calibration=config.get("robot_calibration", {}),
            camera_calibration=config.get("camera_calibration", {}),
            dataset_metadata={
                "format": "bimanual_intermediate_v1",
                "collection_mode": "maniflow_hil_corrections_only",
                "robot_fps": robot_fps,
                "policy_action_hz": policy_action_hz,
                "capture_hz": policy_action_hz,
                "action_bridge_duration_s": action_bridge_duration_s,
                "camera_fps": camera_fps,
                "task_description": str(task_description),
                "maniflow_server": normalized_maniflow_server,
                "maniflow_cameras": list(cameras_for_maniflow),
                "hil_protocol": hil_protocol_cfg.protocol,
                "hil_max_interventions_per_rollout": hil_protocol_cfg.max_interventions_per_rollout,
                "hil_controls": asdict(hil_controls_cfg),
                "camera_configs": [asdict(cfg) for cfg in camera_cfgs],
                "teleop_control": {
                    "startup_alignment": "hold_current",
                    "relative_correction": True,
                    "move_followers_to_home": bool(teleop_value(args, config, "move_followers_to_home", False)),
                    "home_position_file": str(
                        Path(teleop_value(args, config, "home_position_file", default_home_position_file(config))).expanduser()
                    ),
                },
            },
        )
    )

    robot = BimanualRobot.from_lerobot(robot_cfg)
    leader = BimanualLeader.from_lerobot(leader_cfg)
    observation_buffer = LatestObservationBuffer()
    action_buffer = ActionChunkBuffer(
        policy_action_hz=policy_action_hz,
        bridge_duration_s=action_bridge_duration_s,
    )
    armed = threading.Event()
    reset_requested = threading.Event()
    policy_stop_requested = threading.Event()
    generation_gate = HILGenerationGate()
    worker = PolicyWorker(
        runtime=cast(Any, runtime),
        observation_adapter=cast(Any, adapter),
        observation_buffer=observation_buffer,
        action_buffer=action_buffer,
        armed=armed,
        reset_requested=reset_requested,
        stop_requested=policy_stop_requested,
        execution_horizon=int(args.execution_horizon),
        replan_threshold=int(args.replan_threshold),
        chunk_execution_mode=args.chunk_execution_mode,
        wait_timeout_s=float(args.policy_wait_timeout_s),
        generation_provider=generation_gate.current_generation,
        response_accepted=generation_gate.accepts_response,
    )
    flag = StopFlag()
    install_signal_handlers(flag)
    listener = install_keyboard_estop(flag, args.emergency_stop_key)
    max_action_age_s = None if args.disable_action_age_check else float(args.max_action_age_s)

    print("Policy runtime: ManiFlow remote HIL")
    print(f"Task: {task_description}")
    print(f"Correction dataset: {args.output_dir.expanduser()}")
    print(
        f"Policy trajectory: {policy_action_hz:.2f} Hz knots, {robot_fps} Hz robot loop; "
        f"HIL recording: {policy_action_hz:.2f} Hz"
    )
    print(f"Action 0 bridge: {action_bridge_duration_s:.3f} s")
    print(
        "HIL protocol: "
        f"{hil_protocol_cfg.protocol} "
        f"(max interventions per rollout={hil_protocol_cfg.max_interventions_per_rollout})"
    )
    print(
        "HIL controls: "
        f"pause/resume={hil_controls_cfg.pause_resume}, "
        f"correction={hil_controls_cfg.correction}, "
        f"clutch={hil_controls_cfg.clutch}, "
        f"arm={hil_controls_cfg.arm}, reset={hil_controls_cfg.reset}, "
        f"estop='{args.emergency_stop_key}'"
    )
    try:
        cameras.start()
        controls.start()
        leader.connect(calibrate=args.calibrate)
        robot.connect(calibrate=args.calibrate)
        move_followers_home(robot, config, args)
        generation_gate.bump(phase=HILPhase.AUTONOMOUS)
        runtime.reset()
        worker.start()
        run_hil_loop(
            robot=robot,
            leader=leader,
            cameras=cameras,
            controls=controls,
            recorder=recorder,
            policy_worker=worker,
            protocol_config=hil_protocol_cfg,
            generation_gate=generation_gate,
            policy_server=normalized_maniflow_server,
            home_followers=lambda: move_followers_home(robot, config, args),
            observation_buffer=observation_buffer,
            action_buffer=action_buffer,
            armed=armed,
            reset_requested=reset_requested,
            policy_stop_requested=policy_stop_requested,
            image_feature_keys=runtime.image_feature_keys,
            robot_fps=robot_fps,
            capture_hz=policy_action_hz,
            max_action_age_s=max_action_age_s,
            max_action_delta=args.max_action_delta,
            dry_run=bool(args.dry_run),
            flag=flag,
        )
    finally:
        logger.info("Shutting down ManiFlow HIL: %s", flag.reason or "normal exit")
        policy_stop_requested.set()
        worker.stop()
        if recorder.is_recording:
            recorder.discard()
        robot.disconnect()
        leader.disconnect()
        cameras.stop()
        controls.stop()
        if listener is not None:
            listener.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
