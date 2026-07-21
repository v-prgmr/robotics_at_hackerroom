"""Bimanual SO-100 teleoperation and data-recording CLI."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from bimanual_collection.control.clutch import BimanualClutchController, ControlEvent, StartupAlignment
from bimanual_collection.hardware.bimanual_leader import BimanualLeader, BimanualLeaderConfig
from bimanual_collection.hardware.bimanual_robot import BimanualRobot, BimanualRobotConfig, JointLimit
from bimanual_collection.hardware.cameras import CameraConfig, CameraConfigError, MultiCameraManager
from bimanual_collection.hardware.pedals import FootSwitchConfig, FootSwitchManager, PedalSnapshot
from bimanual_collection.recording.episode import TimestepSample, gripper_state
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingControlConfig:
    """Keyboard controls for episode recording lifecycle."""

    manual_start: bool = False
    start_save_key: str = "r"
    cancel_key: str | None = "c"
    status_interval_s: float = 5.0


class RecordingHotkeys:
    """Thread-safe edge flags set by keyboard callbacks and consumed by the control loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_save = False
        self._cancel = False

    def request_start_save(self) -> None:
        with self._lock:
            self._start_save = True

    def request_cancel(self) -> None:
        with self._lock:
            self._cancel = True

    def consume(self) -> tuple[bool, bool]:
        with self._lock:
            start_save = self._start_save
            cancel = self._cancel
            self._start_save = False
            self._cancel = False
        return start_save, cancel


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def cfg_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name.replace("_", "-"), config.get(name, default))


def teleop_control_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("teleop_control", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("teleop_control must be a mapping")
    return raw


def teleop_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    control = teleop_control_config(config)
    return control.get(name.replace("_", "-"), control.get(name, default))


def default_home_position_file(config: dict[str, Any]) -> Path:
    raw = teleop_control_config(config).get("home_position_file", config.get("home_position_file"))
    if raw:
        return Path(raw).expanduser()
    calibration_dir_raw = config.get("calibration_dir")
    calibration_dir = Path(calibration_dir_raw).expanduser() if calibration_dir_raw else Path("calibration/so100_bimanual")
    return calibration_dir / "home_positions.yaml"


def load_follower_home_positions(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Home position file must be a mapping: {path}")
    positions = data.get("home_positions", {})
    if not isinstance(positions, dict):
        raise ValueError(f"home_positions must be a mapping: {path}")

    def role_joints(role: str) -> dict[str, float]:
        entry = positions.get(role)
        if not isinstance(entry, dict) or not isinstance(entry.get("joints"), dict):
            raise ValueError(f"Missing joints for {role} in {path}")
        joints = {key: float(value) for key, value in entry["joints"].items() if key.endswith(".pos")}
        if not joints:
            raise ValueError(f"No '.pos' joints found for {role} in {path}")
        return joints

    return role_joints("left_follower"), role_joints("right_follower")


def move_followers_home(robot: BimanualRobot, config: dict[str, Any], args: argparse.Namespace) -> None:
    enabled = bool(teleop_value(args, config, "move_followers_to_home", False))
    if not enabled:
        return
    home_file = Path(teleop_value(args, config, "home_position_file", default_home_position_file(config))).expanduser()
    left_home, right_home = load_follower_home_positions(home_file)
    duration_s = float(teleop_value(args, config, "home_move_duration_s", 2.0))
    steps = int(teleop_value(args, config, "home_move_steps", 120))
    logger.info("Moving followers to safe home from %s over %.2fs", home_file, duration_s)
    robot.move_to_positions(left_home, right_home, duration_s=duration_s, steps=steps)


def parse_joint_limits(raw_limits: dict[str, Any] | None) -> dict[str, JointLimit]:
    limits: dict[str, JointLimit] = {}
    for key, value in (raw_limits or {}).items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Joint limit for {key} must be [min, max]")
        limits[key] = JointLimit(float(value[0]), float(value[1]))
    return limits


def calibration_id(config: dict[str, Any], role: str, default: str) -> str:
    ids = config.get("calibration_ids", {}) or {}
    if not isinstance(ids, dict):
        raise ValueError("calibration_ids must be a mapping")
    return str(ids.get(role, default))


def build_camera_configs(args: argparse.Namespace, config: dict[str, Any]) -> list[CameraConfig]:
    width = int(cfg_value(args, config, "camera_width", 1280))
    height = int(cfg_value(args, config, "camera_height", 720))
    fps = int(cfg_value(args, config, "camera_fps", 30))
    buffer_size = int(cfg_value(args, config, "camera_buffer_size", max(120, fps * 4)))
    stale_after_s = float(cfg_value(args, config, "camera_stale_after_s", 0.20))
    timeout_s = float(cfg_value(args, config, "camera_timeout_s", 1.0))
    camera_map = {
        "overhead": cfg_value(args, config, "overhead_camera"),
        "left_wrist": cfg_value(args, config, "left_wrist_camera"),
        "right_wrist": cfg_value(args, config, "right_wrist_camera"),
    }
    return [
        CameraConfig(
            name=name,
            device=device,
            width=width,
            height=height,
            fps=fps,
            buffer_size=buffer_size,
            stale_after_s=stale_after_s,
            timeout_s=timeout_s,
        )
        for name, device in camera_map.items()
        if device is not None
    ]


def build_footswitch_config(args: argparse.Namespace, config: dict[str, Any]) -> FootSwitchConfig:
    raw = config.get("footswitch", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("footswitch config must be a mapping")
    keyboard_cfg = raw.get("keyboard", {}) or {}
    evdev_cfg = raw.get("evdev", {}) or {}
    enabled = args.footswitch_enabled if args.footswitch_enabled is not None else bool(raw.get("enabled", False))
    backend = args.footswitch_backend or raw.get("backend", "keyboard")
    debounce_s = args.pedal_debounce_s if args.pedal_debounce_s is not None else float(raw.get("debounce_s", 0.05))
    third_pedal_mode = str(args.footswitch_third_pedal_mode or raw.get("third_pedal_mode", "pause"))
    if third_pedal_mode not in ("pause", "recording"):
        raise ValueError("footswitch.third_pedal_mode must be one of: pause, recording")
    recording_hold_cancel_s = (
        args.footswitch_recording_hold_cancel_s
        if args.footswitch_recording_hold_cancel_s is not None
        else float(raw.get("recording_hold_cancel_s", 1.0))
    )
    if backend == "evdev":
        return FootSwitchConfig(
            enabled=bool(enabled),
            backend="evdev",
            debounce_s=float(debounce_s),
            device=args.footswitch_device or evdev_cfg.get("device"),
            left_clutch=str(args.left_clutch_code or evdev_cfg.get("left_clutch_code", "KEY_1")),
            right_clutch=str(args.right_clutch_code or evdev_cfg.get("right_clutch_code", "KEY_2")),
            pause=str(args.pause_code or evdev_cfg.get("pause_code", "KEY_3")),
            third_pedal_mode=third_pedal_mode,
            recording_hold_cancel_s=recording_hold_cancel_s,
        )
    return FootSwitchConfig(
        enabled=bool(enabled),
        backend="keyboard",
        debounce_s=float(debounce_s),
        left_clutch=str(args.left_clutch_key or keyboard_cfg.get("left_clutch_key", "1")),
        right_clutch=str(args.right_clutch_key or keyboard_cfg.get("right_clutch_key", "2")),
        pause=str(args.pause_key or keyboard_cfg.get("pause_key", "3")),
        third_pedal_mode=third_pedal_mode,
        recording_hold_cancel_s=recording_hold_cancel_s,
    )


def build_recording_control_config(args: argparse.Namespace, config: dict[str, Any]) -> RecordingControlConfig:
    raw = config.get("recording_control", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("recording_control must be a mapping")

    def optional_key(value: Any) -> str | None:
        if value is None:
            return None
        key = str(value).strip()
        if not key or key.lower() in ("none", "null", "disabled"):
            return None
        return key

    manual_start = args.recording_manual_start if args.recording_manual_start is not None else bool(raw.get("manual_start", False))
    status_interval_s = (
        args.recording_status_interval_s
        if args.recording_status_interval_s is not None
        else float(raw.get("status_interval_s", 5.0))
    )
    return RecordingControlConfig(
        manual_start=bool(manual_start),
        start_save_key=str(args.record_start_save_key or raw.get("start_save_key", "r")),
        cancel_key=optional_key(args.record_cancel_key if args.record_cancel_key is not None else raw.get("cancel_key", "c")),
        status_interval_s=float(status_interval_s),
    )


def log_control_events(
    recorder: EpisodeRecorder | None,
    episode_id: str,
    sample_index: int,
    events: list[ControlEvent],
    pedal_state: PedalSnapshot,
) -> None:
    if recorder is None or not events:
        return
    now_mono = time.monotonic()
    now_wall = time.time()
    for event in events:
        recorder.add_event(
            {
                "episode_id": episode_id,
                "timestep_index": sample_index,
                "monotonic_timestamp_s": now_mono,
                "wall_timestamp_s": now_wall,
                "event": event.event,
                "side": event.side,
                "message": event.message,
                "left_clutch_active": pedal_state.left_clutch_active,
                "right_clutch_active": pedal_state.right_clutch_active,
                "recording_paused": pedal_state.recording_paused,
            }
        )


def list_serial_ports() -> None:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found")
        return
    for port in ports:
        print(f"{port.device}\t{port.description}\t{port.hwid}")


def list_camera_devices() -> None:
    print(json.dumps(MultiCameraManager.list_available(), indent=2, default=str))


def print_camera_preflight(cameras: MultiCameraManager, open_cameras: bool = True) -> bool:
    """Print camera preflight results and return whether every camera passed."""

    results = cameras.preflight(open_cameras=open_cameras)
    ok = True
    for result in results:
        print(f"{result.name}:")
        print(f"  configured: {result.configured_device}")
        print(f"  resolved:   {result.resolved_device}")
        if open_cameras:
            print(f"  opened:     {result.opened}")
            print(f"  frame:      {result.captured_frame}")
            if result.width is not None and result.height is not None:
                print(f"  actual:     {result.width}x{result.height}@{result.fps:.2f}")
            if result.backend:
                print(f"  backend:    {result.backend}")
            if result.error:
                print(f"  error:      {result.error}")
                ok = False
    return ok


class StopFlag:
    """Signal-safe stop flag used for Ctrl+C and emergency-stop paths."""

    def __init__(self) -> None:
        self.stop = False
        self.reason = ""

    def request(self, reason: str) -> None:
        self.stop = True
        self.reason = reason


def install_signal_handlers(flag: StopFlag) -> None:
    def _handler(signum, _frame) -> None:
        flag.request(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def install_keyboard_estop(flag: StopFlag, key_name: str) -> Any | None:
    """Start a pynput listener. Returns None when unavailable/headless."""

    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Keyboard listener unavailable: %s", exc)
        return None

    def on_press(key) -> None:
        try:
            name = key.char
        except AttributeError:
            name = str(key).replace("Key.", "")
        if name == key_name:
            flag.request(f"emergency stop key '{key_name}'")

    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    except Exception as exc:
        logger.warning("Keyboard listener failed to start: %s", exc)
        return None


def install_recording_hotkeys(config: RecordingControlConfig, hotkeys: RecordingHotkeys) -> Any | None:
    """Start a keyboard listener for episode start/save and cancel controls."""

    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("Recording hotkeys unavailable: %s", exc)
        print(f"Recording hotkeys unavailable: {exc}")
        return None

    pressed: set[str] = set()

    def normalize(key: Any) -> str:
        try:
            return str(key.char)
        except AttributeError:
            return str(key).replace("Key.", "")

    def on_press(key) -> None:
        name = normalize(key)
        if name in pressed:
            return
        pressed.add(name)
        if name == config.start_save_key:
            hotkeys.request_start_save()
        elif config.cancel_key is not None and name == config.cancel_key:
            hotkeys.request_cancel()

    def on_release(key) -> None:
        pressed.discard(normalize(key))

    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        return listener
    except Exception as exc:
        logger.warning("Recording hotkey listener failed to start: %s", exc)
        print(f"Recording hotkey listener failed to start: {exc}")
        return None


def precise_sleep_until(deadline_s: float) -> None:
    remaining = deadline_s - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def start_recording_episode(recorder: EpisodeRecorder) -> str:
    episode_id = recorder.start()
    print(f"Started recording {recorder.episode_label} -> {recorder.current_final_dir}")
    logger.info("Started recording %s (%s) in %s", recorder.episode_label, episode_id, recorder.output_dir)
    return episode_id


def save_recording_episode(recorder: EpisodeRecorder, reason: str, success: bool | None = None) -> Path:
    episode_label = recorder.episode_label
    episode_id = recorder.episode_id or "unknown"
    samples = recorder.sample_count
    elapsed_s = recorder.elapsed_s
    path = recorder.stop_and_save(success=success, operator_notes=reason)
    print(f"Saved recording {episode_label}: {samples} samples, {elapsed_s:.1f}s -> {path}")
    logger.info("Saved recording %s (%s) with %d samples to %s", episode_label, episode_id, samples, path)
    return path


def cancel_recording_episode(recorder: EpisodeRecorder) -> None:
    episode_label = recorder.episode_label
    episode_id = recorder.episode_id or "unknown"
    samples = recorder.sample_count
    elapsed_s = recorder.elapsed_s
    recorder.discard()
    print(f"Cancelled recording {episode_label}: discarded {samples} samples after {elapsed_s:.1f}s; recording is idle")
    logger.info("Cancelled recording %s (%s) after %d samples", episode_label, episode_id, samples)


def apply_recording_request(
    recorder: EpisodeRecorder,
    *,
    start_save_requested: bool,
    cancel_requested: bool,
) -> str | None:
    """Apply one recording lifecycle request and return the active episode id, if any."""

    if cancel_requested:
        if recorder.is_recording:
            cancel_recording_episode(recorder)
        else:
            print("Cancel ignored: recording is already idle")
        return None

    if not start_save_requested:
        return recorder.episode_id

    if recorder.is_recording:
        save_recording_episode(recorder, reason="Saved by recording control")
        print("Recording is idle. Press the third pedal or start/save key to start another episode.")
        return None

    return start_recording_episode(recorder)


def run_control_loop(
    robot: BimanualRobot,
    leader: BimanualLeader,
    cameras: MultiCameraManager,
    recorder: EpisodeRecorder | None,
    footswitch: FootSwitchManager,
    robot_fps: int,
    startup_alignment: StartupAlignment,
    recording_control: RecordingControlConfig,
    recording_hotkeys: RecordingHotkeys,
    flag: StopFlag,
) -> None:
    period_s = 1.0 / robot_fps
    sample_index = 0
    previous_loop_start = time.monotonic()
    episode_id = "teleop-idle"
    if recorder is not None:
        print(f"Dataset directory: {recorder.output_dir}")
        cancel_label = recording_control.cancel_key if recording_control.cancel_key is not None else "disabled"
        print(
            "Recording controls: "
            f"'{recording_control.start_save_key}' start/save, "
            f"keyboard cancel {cancel_label}, third pedal start/save or hold-cancel, 'q' quit"
        )
        if recording_control.manual_start:
            print("Recording is idle. Press the third pedal or start/save key to start episode recording.")
        else:
            episode_id = start_recording_episode(recorder)
    controller = BimanualClutchController(startup_alignment=startup_alignment)
    last_recording_status_s = time.monotonic()

    while not flag.stop:
        loop_start = time.monotonic()
        wall_time = time.time()
        try:
            pedal_state = footswitch.snapshot()
            leader_state = leader.read()
            follower_state = robot.read_state()
            if pedal_state.failed:
                control = controller.update(
                    leader_state.left,
                    leader_state.right,
                    follower_state.left,
                    follower_state.right,
                    left_clutch_active=True,
                    right_clutch_active=True,
                    recording_paused=True,
                )
                robot.send_actions(control.left_action, control.right_action)
                flag.request(f"footswitch failure: {pedal_state.error}")
                logger.error("Footswitch failed; holding both arms and stopping")
                break
            control = controller.update(
                leader_state.left,
                leader_state.right,
                follower_state.left,
                follower_state.right,
                left_clutch_active=pedal_state.left_clutch_active,
                right_clutch_active=pedal_state.right_clutch_active,
                recording_paused=pedal_state.recording_paused,
            )
            command = robot.send_actions(control.left_action, control.right_action)
        except Exception as exc:
            flag.request(f"hardware failure: {exc}")
            logger.exception("Hardware failure; stopping both arms")
            break

        start_save_requested, cancel_requested = recording_hotkeys.consume()
        start_save_requested = start_save_requested or pedal_state.recording_start_save_edge
        cancel_requested = cancel_requested or pedal_state.recording_cancel_edge
        if recorder is not None:
            previous_recording = recorder.is_recording
            active_episode_id = apply_recording_request(
                recorder,
                start_save_requested=start_save_requested,
                cancel_requested=cancel_requested,
            )
            if previous_recording != recorder.is_recording or start_save_requested or cancel_requested:
                episode_id = active_episode_id or "teleop-idle"
                sample_index = 0
                last_recording_status_s = time.monotonic()

        sample_timestamp = time.monotonic()
        camera_sample = cameras.match(sample_timestamp)
        loop_duration_s = time.monotonic() - loop_start
        interval_s = loop_start - previous_loop_start
        previous_loop_start = loop_start
        measured_hz = 1.0 / interval_s if interval_s > 0 else 0.0
        log_control_events(recorder, episode_id, sample_index, control.events, pedal_state)

        sample = TimestepSample(
            episode_id=episode_id,
            timestep_index=sample_index,
            monotonic_timestamp_s=sample_timestamp,
            wall_timestamp_s=wall_time,
            left_leader_joints=leader_state.left,
            right_leader_joints=leader_state.right,
            left_follower_joints=follower_state.left,
            right_follower_joints=follower_state.right,
            left_gripper_state=gripper_state(follower_state.left),
            right_gripper_state=gripper_state(follower_state.right),
            left_commanded_action=command.left,
            right_commanded_action=command.right,
            camera_matches=camera_sample.matches,
            measured_control_hz=measured_hz,
            loop_duration_s=loop_duration_s,
            metadata={
                "leader_read_duration_s": leader_state.read_finished_monotonic_s
                - leader_state.read_started_monotonic_s,
                "follower_read_duration_s": follower_state.read_finished_monotonic_s
                - follower_state.read_started_monotonic_s,
                "command_duration_s": command.command_finished_monotonic_s - command.command_started_monotonic_s,
                "left_clutch_active": control.left_clutch_active,
                "right_clutch_active": control.right_clutch_active,
                "recording_paused": control.recording_paused,
            },
        )
        if recorder is not None and control.should_record:
            recorder.add_sample(sample)
            if recorder.is_recording:
                sample_index += 1
                now_s = time.monotonic()
                if recording_control.status_interval_s > 0 and now_s - last_recording_status_s >= recording_control.status_interval_s:
                    print(
                        f"Recording {recorder.episode_label}: "
                        f"{recorder.sample_count} samples, {recorder.elapsed_s:.1f}s -> {recorder.current_final_dir}"
                    )
                    last_recording_status_s = now_s

        precise_sleep_until(loop_start + period_s)

    if not flag.reason.startswith("hardware failure") and robot.is_connected:
        left_hold = controller.left.last_commanded_action
        right_hold = controller.right.last_commanded_action
        if left_hold is not None and right_hold is not None:
            with contextlib.suppress(Exception):
                robot.send_actions(left_hold, right_hold)

    if recorder is not None and recorder.is_recording:
        if flag.reason.startswith("hardware failure"):
            cancel_recording_episode(recorder)
        else:
            save_recording_episode(recorder, reason=f"Stopped by {flag.reason or 'operator'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--left-robot-port")
    parser.add_argument("--right-robot-port")
    parser.add_argument("--left-leader-port")
    parser.add_argument("--right-leader-port")
    parser.add_argument("--overhead-camera")
    parser.add_argument("--left-wrist-camera")
    parser.add_argument("--right-wrist-camera")
    parser.add_argument("--robot-fps", type=int)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-buffer-size", type=int)
    parser.add_argument("--camera-stale-after-s", type=float)
    parser.add_argument("--camera-timeout-s", type=float)
    parser.add_argument("--startup-alignment", choices=["hold_current", "leader_absolute"])
    parser.add_argument("--move-followers-to-home", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--home-position-file", type=Path)
    parser.add_argument("--home-move-duration-s", type=float)
    parser.add_argument("--home-move-steps", type=int)
    parser.add_argument("--recording-manual-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record-start-save-key")
    parser.add_argument("--record-cancel-key")
    parser.add_argument("--recording-status-interval-s", type=float)
    parser.add_argument("--footswitch-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--footswitch-backend", choices=["keyboard", "evdev"])
    parser.add_argument("--footswitch-device")
    parser.add_argument("--left-clutch-key")
    parser.add_argument("--right-clutch-key")
    parser.add_argument("--pause-key")
    parser.add_argument("--left-clutch-code")
    parser.add_argument("--right-clutch-code")
    parser.add_argument("--pause-code")
    parser.add_argument("--pedal-debounce-s", type=float)
    parser.add_argument("--footswitch-third-pedal-mode", choices=["pause", "recording"])
    parser.add_argument("--footswitch-recording-hold-cancel-s", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-start-number", type=int, help="First episode number for this recording run, e.g. 51 -> episode-000051.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-description")
    parser.add_argument("--warmup-s", type=float)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--emergency-stop-key", default="q")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--check-cameras", action="store_true", help="Validate configured cameras and exit.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.list_ports:
        list_serial_ports()
        return
    if args.list_cameras:
        list_camera_devices()
        return

    config = load_config(args.config)
    required = ["left_robot_port", "right_robot_port", "left_leader_port", "right_leader_port"]
    missing = [name for name in required if cfg_value(args, config, name) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    robot_fps = int(cfg_value(args, config, "robot_fps", 60))
    camera_fps = int(cfg_value(args, config, "camera_fps", 30))
    output_dir = Path(cfg_value(args, config, "output_dir", "./data/bimanual"))
    episode_start_number = cfg_value(args, config, "episode_start_number")
    episode_start_number = int(episode_start_number) if episode_start_number is not None else None
    if episode_start_number is not None and episode_start_number < 1:
        parser.error("episode_start_number must be >= 1")
    record = bool(args.record or config.get("record", False))
    startup_alignment = str(teleop_value(args, config, "startup_alignment", "hold_current"))
    if startup_alignment not in ("hold_current", "leader_absolute"):
        parser.error("startup_alignment must be one of: hold_current, leader_absolute")

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
    camera_cfgs = build_camera_configs(args, config)
    footswitch_cfg = build_footswitch_config(args, config)
    recording_control_cfg = build_recording_control_config(args, config)
    if recording_control_cfg.cancel_key is not None and recording_control_cfg.start_save_key == recording_control_cfg.cancel_key:
        parser.error("record_start_save_key and record_cancel_key must be different")
    recording_keys = [recording_control_cfg.start_save_key]
    if recording_control_cfg.cancel_key is not None:
        recording_keys.append(recording_control_cfg.cancel_key)
    if args.emergency_stop_key in recording_keys:
        parser.error("recording hotkeys must not match the emergency stop key")

    try:
        cameras = MultiCameraManager(camera_cfgs)
    except CameraConfigError as exc:
        parser.error(str(exc))

    if args.check_cameras:
        raise SystemExit(0 if print_camera_preflight(cameras, open_cameras=True) else 1)

    robot = BimanualRobot.from_lerobot(robot_cfg)
    leader = BimanualLeader.from_lerobot(leader_cfg)
    footswitch = FootSwitchManager(footswitch_cfg)
    recorder = None
    if record:
        recorder = EpisodeRecorder(
            RecorderConfig(
                output_dir=output_dir,
                warmup_s=float(cfg_value(args, config, "warmup_s", 0.0)),
                task_description=cfg_value(args, config, "task_description", ""),
                camera_fps=camera_fps,
                episode_start_number=episode_start_number,
                environment=config.get("environment", {}),
                robot_calibration=config.get("robot_calibration", {}),
                camera_calibration=config.get("camera_calibration", {}),
                dataset_metadata={
                    "format": "bimanual_intermediate_v1",
                    "robot_fps": robot_fps,
                    "camera_fps": camera_fps,
                    "teleop_control": {
                        "startup_alignment": startup_alignment,
                        "move_followers_to_home": bool(teleop_value(args, config, "move_followers_to_home", False)),
                        "home_position_file": str(
                            Path(teleop_value(args, config, "home_position_file", default_home_position_file(config))).expanduser()
                        ),
                    },
                    "recording_control": asdict(recording_control_cfg),
                    "episode_start_number": episode_start_number,
                    "camera_configs": [asdict(cfg) for cfg in camera_cfgs],
                    "footswitch_config": asdict(footswitch_cfg),
                },
            )
        )

    flag = StopFlag()
    recording_hotkeys = RecordingHotkeys()
    install_signal_handlers(flag)
    listener = install_keyboard_estop(flag, args.emergency_stop_key)
    recording_listener = install_recording_hotkeys(recording_control_cfg, recording_hotkeys) if record else None

    try:
        cameras.start()
        footswitch.start()
        leader.connect(calibrate=args.calibrate)
        robot.connect(calibrate=args.calibrate)
        move_followers_home(robot, config, args)
        run_control_loop(
            robot,
            leader,
            cameras,
            recorder,
            footswitch,
            robot_fps,
            startup_alignment,
            recording_control_cfg,
            recording_hotkeys,
            flag,
        )
    finally:
        logger.info("Shutting down: %s", flag.reason or "normal exit")
        if recorder is not None and recorder.is_recording:
            recorder.discard()
        robot.disconnect()
        leader.disconnect()
        cameras.stop()
        footswitch.stop()
        if listener is not None:
            listener.stop()
        if recording_listener is not None:
            recording_listener.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
