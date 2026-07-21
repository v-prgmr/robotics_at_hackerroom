"""Asynchronous multi-camera capture with timestamp matching."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for one RGB camera stream."""

    name: str
    device: str | int
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 120
    stale_after_s: float = 0.20
    timeout_s: float = 1.0
    fourcc: str | None = "MJPG"


@dataclass(frozen=True)
class CameraFrame:
    """Timestamped camera frame in host monotonic time."""

    camera_name: str
    frame_index: int
    image: np.ndarray
    camera_timestamp_s: float
    host_timestamp_s: float
    hardware_timestamp_s: float | None = None


@dataclass(frozen=True)
class MatchedCameraFrame:
    """Result of matching a camera frame to one robot timestep."""

    camera_name: str
    frame: CameraFrame | None
    frame_index: int | None
    camera_timestamp_s: float | None
    host_timestamp_s: float | None
    hardware_timestamp_s: float | None
    age_s: float | None
    stale: bool
    missing: bool
    dropped_frames: int
    disconnected: bool


@dataclass(frozen=True)
class CameraPreflightResult:
    """Resolved camera identity and optional open/read check result."""

    name: str
    configured_device: str | int
    resolved_device: str | int
    opened: bool | None = None
    captured_frame: bool | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    backend: str | None = None
    error: str | None = None


class CameraConfigError(ValueError):
    """Raised when camera config is invalid before hardware startup."""


def resolve_camera_device(device: str | int) -> str | int:
    """Resolve device paths to physical `/dev/video*` targets when possible."""

    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    path = Path(device).expanduser()
    if path.exists():
        return str(path.resolve())
    return str(path)


def cv2_device_arg(device: str | int) -> str | int:
    """Convert config device into an argument suitable for OpenCV/LeRobot."""

    if isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    return str(Path(device).expanduser())


class AsyncCameraStream:
    """Capture frames in a bounded background buffer.

    The robot loop calls `match_nearest()` without blocking. Frames are dropped
    explicitly when the ring buffer is full, and the dropped count is reported in
    the per-timestep camera metadata.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._buffer: deque[CameraFrame] = deque(maxlen=config.buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Any | None = None
        self._frame_index = 0
        self._dropped_frames = 0
        self._last_error: str | None = None
        self._disconnected = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    @property
    def disconnected(self) -> bool:
        with self._lock:
            return self._disconnected

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._connect_camera()
        self._thread = threading.Thread(target=self._capture_loop, name=f"camera-{self.config.name}", daemon=True)
        self._thread.start()
        logger.info("Started camera stream %s", self.config.name)

    def _connect_camera(self) -> None:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        device = cv2_device_arg(self.config.device)
        self._camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=Path(device) if isinstance(device, str) and device.startswith("/") else device,
                fps=self.config.fps,
                width=self.config.width,
                height=self.config.height,
                fourcc=self.config.fourcc,
            )
        )
        self._camera.connect(warmup=True)

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                image = self._camera.read_latest(max_age_ms=int(self.config.timeout_s * 1000))
                now = time.monotonic()
                frame = CameraFrame(
                    camera_name=self.config.name,
                    frame_index=self._frame_index,
                    image=image,
                    camera_timestamp_s=now,
                    host_timestamp_s=now,
                    hardware_timestamp_s=getattr(self._camera, "latest_timestamp", None),
                )
                with self._lock:
                    if len(self._buffer) == self._buffer.maxlen:
                        self._dropped_frames += 1
                    self._buffer.append(frame)
                    self._frame_index += 1
                    self._disconnected = False
                    self._last_error = None
                time.sleep(max(0.0, 0.5 / self.config.fps))
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._disconnected = True
                logger.warning("Camera %s capture error: %s", self.config.name, exc)
                time.sleep(min(self.config.timeout_s, 1.0))

    def match_nearest(self, timestamp_s: float) -> MatchedCameraFrame:
        """Return the nearest buffered frame to a robot timestamp."""

        with self._lock:
            frames = list(self._buffer)
            dropped = self._dropped_frames
            disconnected = self._disconnected
        if not frames:
            return MatchedCameraFrame(
                camera_name=self.config.name,
                frame=None,
                frame_index=None,
                camera_timestamp_s=None,
                host_timestamp_s=None,
                hardware_timestamp_s=None,
                age_s=None,
                stale=True,
                missing=True,
                dropped_frames=dropped,
                disconnected=disconnected,
            )
        frame = min(frames, key=lambda item: abs(item.host_timestamp_s - timestamp_s))
        age_s = timestamp_s - frame.host_timestamp_s
        stale = abs(age_s) > self.config.stale_after_s
        return MatchedCameraFrame(
            camera_name=self.config.name,
            frame=frame,
            frame_index=frame.frame_index,
            camera_timestamp_s=frame.camera_timestamp_s,
            host_timestamp_s=frame.host_timestamp_s,
            hardware_timestamp_s=frame.hardware_timestamp_s,
            age_s=age_s,
            stale=stale,
            missing=False,
            dropped_frames=dropped,
            disconnected=disconnected,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._camera is not None:
            try:
                if self._camera.is_connected:
                    self._camera.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting camera %s: %s", self.config.name, exc)
            self._camera = None


@dataclass(frozen=True)
class MultiCameraSample:
    """Camera matches for one robot timestep."""

    timestamp_s: float
    matches: dict[str, MatchedCameraFrame]


@dataclass
class MultiCameraManager:
    """Manages overhead and wrist camera streams with nonblocking matching."""

    cameras: list[CameraConfig]
    streams: dict[str, AsyncCameraStream] = field(init=False)

    def __post_init__(self) -> None:
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"Camera names must be unique: {names}")
        self.validate_unique_devices(self.cameras)
        self.streams = {camera.name: AsyncCameraStream(camera) for camera in self.cameras}

    def start(self) -> None:
        self.validate_unique_devices(self.cameras)
        started: list[AsyncCameraStream] = []
        try:
            for stream in self.streams.values():
                stream.start()
                started.append(stream)
        except Exception as exc:
            for stream in started:
                stream.stop()
            raise RuntimeError(f"Failed to start camera streams: {exc}") from exc

    def preflight(self, open_cameras: bool = False) -> list[CameraPreflightResult]:
        """Validate camera identity and optionally open/capture every camera."""

        self.validate_unique_devices(self.cameras)
        if not open_cameras:
            return [
                CameraPreflightResult(
                    name=camera.name,
                    configured_device=camera.device,
                    resolved_device=resolve_camera_device(camera.device),
                )
                for camera in self.cameras
            ]

        import cv2  # type: ignore

        results: list[CameraPreflightResult] = []
        for camera in self.cameras:
            capture = None
            try:
                device_arg = cv2_device_arg(camera.device)
                capture = cv2.VideoCapture(device_arg)
                opened = bool(capture.isOpened())
                if not opened:
                    results.append(
                        CameraPreflightResult(
                            name=camera.name,
                            configured_device=camera.device,
                            resolved_device=resolve_camera_device(camera.device),
                            opened=False,
                            captured_frame=False,
                            error="OpenCV could not open device",
                        )
                    )
                    continue
                if camera.fourcc:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera.fourcc))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera.width))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera.height))
                capture.set(cv2.CAP_PROP_FPS, float(camera.fps))
                ok, _frame = capture.read()
                results.append(
                    CameraPreflightResult(
                        name=camera.name,
                        configured_device=camera.device,
                        resolved_device=resolve_camera_device(camera.device),
                        opened=True,
                        captured_frame=bool(ok),
                        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        fps=float(capture.get(cv2.CAP_PROP_FPS)),
                        backend=capture.getBackendName() if hasattr(capture, "getBackendName") else None,
                        error=None if ok else "Device opened but no frame was captured",
                    )
                )
            except Exception as exc:
                results.append(
                    CameraPreflightResult(
                        name=camera.name,
                        configured_device=camera.device,
                        resolved_device=resolve_camera_device(camera.device),
                        opened=False,
                        captured_frame=False,
                        error=str(exc),
                    )
                )
            finally:
                if capture is not None:
                    capture.release()
        return results

    def match(self, timestamp_s: float) -> MultiCameraSample:
        return MultiCameraSample(
            timestamp_s=timestamp_s,
            matches={name: stream.match_nearest(timestamp_s) for name, stream in self.streams.items()},
        )

    def stop(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    @staticmethod
    def validate_unique_devices(cameras: list[CameraConfig]) -> None:
        """Reject duplicate physical devices before OpenCV startup."""

        resolved_to_names: dict[str | int, list[str]] = {}
        missing: list[str] = []
        for camera in cameras:
            if isinstance(camera.device, str) and not camera.device.isdigit():
                path = Path(camera.device).expanduser()
                if camera.device.startswith("/") and not path.exists():
                    missing.append(f"{camera.name}: {camera.device}")
            resolved = resolve_camera_device(camera.device)
            resolved_to_names.setdefault(resolved, []).append(camera.name)
        duplicates = {resolved: names for resolved, names in resolved_to_names.items() if len(names) > 1}
        if missing or duplicates:
            messages: list[str] = []
            if missing:
                messages.append("Missing camera device paths:\n" + "\n".join(f"  {item}" for item in missing))
            if duplicates:
                duplicate_lines = []
                for resolved, names in duplicates.items():
                    duplicate_lines.append(f"  {', '.join(names)} all resolve to {resolved}")
                messages.append(
                    "Duplicate physical camera devices configured:\n"
                    + "\n".join(duplicate_lines)
                    + "\nUse distinct /dev/v4l/by-path/...-video-index0 paths for identical USB cameras."
                )
            raise CameraConfigError("\n".join(messages))

    @staticmethod
    def list_available() -> list[dict[str, Any]]:
        """List OpenCV cameras using LeRobot's camera discovery API."""

        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

        return OpenCVCamera.find_cameras()
