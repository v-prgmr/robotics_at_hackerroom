"""Small PyAV video writer used by the intermediate dataset backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodedFrameInfo:
    """Metadata for a frame appended to a camera MP4."""

    video_frame_index: int
    camera_frame_index: int
    camera_timestamp_s: float
    host_timestamp_s: float


class VideoWriter:
    """Append RGB frames to one MP4 file at a fixed nominal FPS."""

    def __init__(self, path: Path, fps: int, codec: str = "libx264", crf: int = 23) -> None:
        self.path = path
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self._container: Any | None = None
        self._stream: Any | None = None
        self._count = 0

    @property
    def frame_count(self) -> int:
        return self._count

    def append(self, frame: np.ndarray) -> int:
        """Append one HWC RGB/BGR frame and return its video frame index."""

        import av

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HWC 3-channel frame, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self._container is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._container = av.open(str(self.path), "w")
            self._stream = self._container.add_stream(self.codec, rate=self.fps, options={"crf": str(self.crf)})
            self._stream.width = int(frame.shape[1])
            self._stream.height = int(frame.shape[0])
            self._stream.pix_fmt = "yuv420p"

        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self._count
        packet = self._stream.encode(video_frame)
        if packet:
            self._container.mux(packet)
        index = self._count
        self._count += 1
        return index

    def close(self) -> None:
        if self._container is None:
            return
        packet = self._stream.encode()
        if packet:
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
