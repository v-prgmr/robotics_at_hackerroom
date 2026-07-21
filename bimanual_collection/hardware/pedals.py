"""Foot-switch input handling for bimanual teleoperation."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PedalBackend = Literal["keyboard", "evdev"]
ThirdPedalMode = Literal["pause", "recording"]


@dataclass(frozen=True)
class FootSwitchConfig:
    """Configuration for a three-pedal foot switch."""

    enabled: bool = False
    backend: PedalBackend = "keyboard"
    debounce_s: float = 0.05
    device: str | None = None
    left_clutch: str = "1"
    right_clutch: str = "2"
    pause: str = "3"
    third_pedal_mode: ThirdPedalMode = "pause"
    recording_hold_cancel_s: float = 1.0


@dataclass(frozen=True)
class PedalSnapshot:
    """Thread-safe snapshot of current foot-switch state."""

    left_clutch_active: bool
    right_clutch_active: bool
    recording_paused: bool
    pause_edge: bool
    recording_start_save_edge: bool
    recording_cancel_edge: bool
    failed: bool
    error: str | None


class FootSwitchManager:
    """Nonblocking foot-switch manager with debouncing and edge detection."""

    def __init__(self, config: FootSwitchConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._left = False
        self._right = False
        self._paused = False
        self._pause_edge = False
        self._pause_pressed = False
        self._recording_pressed = False
        self._recording_cancel_emitted = False
        self._recording_start_save_edge = False
        self._recording_cancel_edge = False
        self._recording_timer: threading.Timer | None = None
        self._failed = False
        self._error: str | None = None
        self._last_change: dict[str, float] = {"left": 0.0, "right": 0.0, "pause": 0.0, "recording": 0.0}
        self._listener: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self.config.backend == "keyboard":
            self._start_keyboard()
        elif self.config.backend == "evdev":
            self._start_evdev()
        else:
            raise ValueError(f"Unsupported footswitch backend: {self.config.backend}")

    def snapshot(self) -> PedalSnapshot:
        """Return current state and consume the pause rising-edge flag."""

        with self._lock:
            pause_edge = self._pause_edge
            recording_start_save_edge = self._recording_start_save_edge
            recording_cancel_edge = self._recording_cancel_edge
            self._pause_edge = False
            self._recording_start_save_edge = False
            self._recording_cancel_edge = False
            return PedalSnapshot(
                left_clutch_active=self._left,
                right_clutch_active=self._right,
                recording_paused=self._paused,
                pause_edge=pause_edge,
                recording_start_save_edge=recording_start_save_edge,
                recording_cancel_edge=recording_cancel_edge,
                failed=self._failed,
                error=self._error,
            )

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as exc:
                logger.debug("Error stopping keyboard footswitch listener: %s", exc)
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._recording_timer is not None:
            self._recording_timer.cancel()
            self._recording_timer = None

    def _mark_failed(self, message: str) -> None:
        with self._lock:
            self._failed = True
            self._error = message
        logger.error("Footswitch input failed: %s", message)

    def _debounced(self, control: str) -> bool:
        now = time.monotonic()
        if now - self._last_change[control] < self.config.debounce_s:
            return False
        self._last_change[control] = now
        return True

    def _set_clutch(self, side: Literal["left", "right"], active: bool) -> None:
        if not self._debounced(side):
            return
        with self._lock:
            if side == "left":
                changed = self._left != active
                self._left = active
            else:
                changed = self._right != active
                self._right = active
        if changed:
            logger.info("%s clutch %s", side, "engaged" if active else "released")
            print(f"{side.capitalize()} clutch {'engaged' if active else 'released'}")

    def _toggle_pause(self) -> None:
        if not self._debounced("pause"):
            return
        with self._lock:
            self._paused = not self._paused
            self._pause_edge = True
            paused = self._paused
        logger.info("Recording %s", "paused" if paused else "resumed")
        print(f"Recording {'paused' if paused else 'resumed'}")

    def _recording_hold_cancel(self) -> None:
        with self._lock:
            if not self._recording_pressed or self._recording_cancel_emitted:
                return
            self._recording_cancel_emitted = True
            self._recording_cancel_edge = True
        logger.info("Recording cancel requested by held third pedal")
        print("Recording cancel requested")

    def _set_recording_control(self, pressed: bool) -> None:
        if pressed:
            if not self._debounced("recording"):
                return
            with self._lock:
                if self._recording_pressed:
                    return
                self._recording_pressed = True
                self._recording_cancel_emitted = False
            timer = threading.Timer(max(0.0, self.config.recording_hold_cancel_s), self._recording_hold_cancel)
            timer.daemon = True
            with self._lock:
                self._recording_timer = timer
            timer.start()
            return

        with self._lock:
            if not self._recording_pressed:
                return
            self._recording_pressed = False
            cancel_emitted = self._recording_cancel_emitted
            timer = self._recording_timer
            self._recording_timer = None
        if timer is not None:
            timer.cancel()
        if not cancel_emitted:
            with self._lock:
                self._recording_start_save_edge = True
            logger.info("Recording start/save requested by third pedal")
            print("Recording start/save requested")

    def _normalize_keyboard_key(self, key: Any) -> str:
        try:
            return str(key.char)
        except AttributeError:
            return str(key).replace("Key.", "")

    def _start_keyboard(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            self._mark_failed(f"pynput unavailable: {exc}")
            return

        def on_press(key) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.left_clutch:
                self._set_clutch("left", True)
            elif name == self.config.right_clutch:
                self._set_clutch("right", True)
            elif name == self.config.pause:
                if self.config.third_pedal_mode == "recording":
                    self._set_recording_control(True)
                else:
                    with self._lock:
                        if self._pause_pressed:
                            return
                        self._pause_pressed = True
                    self._toggle_pause()

        def on_release(key) -> None:
            name = self._normalize_keyboard_key(key)
            if name == self.config.left_clutch:
                self._set_clutch("left", False)
            elif name == self.config.right_clutch:
                self._set_clutch("right", False)
            elif name == self.config.pause:
                if self.config.third_pedal_mode == "recording":
                    self._set_recording_control(False)
                else:
                    with self._lock:
                        self._pause_pressed = False

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
        except Exception as exc:
            self._mark_failed(f"keyboard listener failed: {exc}")

    def _start_evdev(self) -> None:
        if not self.config.device:
            self._mark_failed("evdev backend requires footswitch.device")
            return
        self._thread = threading.Thread(target=self._evdev_loop, name="footswitch-evdev", daemon=True)
        self._thread.start()

    def _evdev_loop(self) -> None:
        try:
            from evdev import InputDevice, categorize, ecodes
        except Exception as exc:
            self._mark_failed(f"evdev unavailable: {exc}")
            return

        try:
            if self.config.device is None:
                self._mark_failed("evdev backend requires footswitch.device")
                return

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
            left_code = resolve_key_code(self.config.left_clutch)
            right_code = resolve_key_code(self.config.right_clutch)
            pause_code = resolve_key_code(self.config.pause)
            for event in device.read_loop():
                if self._stop.is_set():
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = categorize(event)
                code = key_event.scancode
                if code == left_code:
                    if key_event.keystate == key_event.key_down:
                        self._set_clutch("left", True)
                    elif key_event.keystate == key_event.key_up:
                        self._set_clutch("left", False)
                elif code == right_code:
                    if key_event.keystate == key_event.key_down:
                        self._set_clutch("right", True)
                    elif key_event.keystate == key_event.key_up:
                        self._set_clutch("right", False)
                elif code == pause_code and key_event.keystate == key_event.key_down:
                    if self.config.third_pedal_mode == "recording":
                        self._set_recording_control(True)
                    else:
                        self._toggle_pause()
                elif code == pause_code and key_event.keystate == key_event.key_up and self.config.third_pedal_mode == "recording":
                    self._set_recording_control(False)
        except Exception as exc:
            if not self._stop.is_set():
                self._mark_failed(str(exc))
