from collections import deque

import numpy as np

from bimanual_collection.hardware.cameras import AsyncCameraStream, CameraConfig, CameraFrame


def test_nearest_frame_matching_marks_age_and_stale():
    stream = AsyncCameraStream(CameraConfig(name="overhead", device="/dev/video0", stale_after_s=0.05))
    frames = [
        CameraFrame("overhead", 0, np.zeros((2, 2, 3), dtype=np.uint8), 1.00, 1.00),
        CameraFrame("overhead", 1, np.zeros((2, 2, 3), dtype=np.uint8), 1.03, 1.03),
        CameraFrame("overhead", 2, np.zeros((2, 2, 3), dtype=np.uint8), 1.20, 1.20),
    ]
    with stream._lock:
        stream._buffer = deque(frames, maxlen=3)

    match = stream.match_nearest(1.04)
    assert match.frame_index == 1
    assert abs(match.age_s - 0.01) < 1e-9
    assert not match.stale
    assert not match.missing

    stale = stream.match_nearest(1.12)
    assert stale.stale


def test_missing_frame_reports_disconnected_and_dropped_count():
    stream = AsyncCameraStream(CameraConfig(name="left_wrist", device="/dev/video2"))
    with stream._lock:
        stream._dropped_frames = 4
        stream._disconnected = True

    match = stream.match_nearest(10.0)
    assert match.missing
    assert match.stale
    assert match.dropped_frames == 4
    assert match.disconnected
