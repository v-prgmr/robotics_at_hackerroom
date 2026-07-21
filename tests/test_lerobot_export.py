import numpy as np
import pandas as pd
import pytest

from bimanual_collection.recording.backends.lerobot_export import SequentialVideoFrameReader, _build_lerobot_frame


def test_build_lerobot_frame_concatenates_bimanual_vectors_without_timestamp():
    row = pd.Series(
        {
            "left_follower_joints": np.arange(6, dtype=np.float32),
            "right_follower_joints": np.arange(10, 16, dtype=np.float32),
            "left_commanded_action": np.arange(20, 26, dtype=np.float32),
            "right_commanded_action": np.arange(30, 36, dtype=np.float32),
            "monotonic_timestamp_s": 123.0,
        }
    )

    frame = _build_lerobot_frame(row, task="test task")

    assert "timestamp" not in frame
    assert frame["task"] == "test task"
    np.testing.assert_array_equal(
        frame["observation.state"],
        np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        frame["action"],
        np.asarray([20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35], dtype=np.float32),
    )


def test_sequential_video_frame_reader_reads_without_full_cache(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "frames.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (4, 4))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter mp4v unavailable")
    for value in (20, 80, 140):
        writer.write(np.full((4, 4, 3), value, dtype=np.uint8))
    writer.release()

    reader = SequentialVideoFrameReader(video_path)
    try:
        first = reader.get(0)
        repeated = reader.get(0)
        third = reader.get(2)
    finally:
        reader.close()

    assert first.shape == (4, 4, 3)
    assert repeated is first
    assert third.shape == (4, 4, 3)
    assert float(third.mean()) > float(first.mean())
