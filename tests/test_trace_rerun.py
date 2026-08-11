import json

import numpy as np
import pytest
from PIL import Image

from bimanual_collection.tools import trace_rerun


class FakeRerun:
    class Image:
        def __init__(self, image):
            self.image = image

    def __init__(self):
        self.times = []
        self.logs = []

    def set_time_sequence(self, name, value):
        self.times.append(("sequence", name, value))

    def set_time_seconds(self, name, value):
        self.times.append(("seconds", name, value))

    def log(self, entity, value):
        self.logs.append((entity, value))


def write_trace_run(path, *, sequence=1, camera_name="overhead"):
    images_dir = path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"obs_{sequence:06d}_{camera_name}.png"
    Image.fromarray(np.full((12, 16, 3), sequence, dtype=np.uint8)).save(image_path)
    event = {
        "event": "policy_observation",
        "observation_sequence": sequence,
        "observation_timestamp_s": 12.5,
        "wall_timestamp_s": 100.0,
        "cameras": [
            {
                "camera_name": camera_name,
                "image_path": str(image_path.relative_to(path)),
            }
        ],
    }
    with (path / "events.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def test_discover_trace_dirs_accepts_parent_dir(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_trace_run(run_b)
    write_trace_run(run_a)

    assert trace_rerun.discover_trace_dirs(tmp_path) == [run_a, run_b]


def test_replay_trace_images_logs_image_with_timeline(tmp_path):
    run = tmp_path / "run-a"
    write_trace_run(run, sequence=7, camera_name="left_wrist")
    rr = FakeRerun()

    count = trace_rerun.replay_trace_images(rr, run, run_index=2, namespace="runs/run-a/")

    assert count == 1
    assert rr.logs[0][0] == "runs/run-a/cameras/left_wrist/image"
    assert rr.logs[0][1].image.shape == (12, 16, 3)
    assert ("sequence", "trace_run", 2) in rr.times
    assert ("sequence", "observation_sequence", 7) in rr.times
    assert ("seconds", "trace_time", 12.5) in rr.times


def test_export_trace_video_writes_readable_mp4(tmp_path):
    cv2 = pytest.importorskip("cv2")
    run = tmp_path / "run-a"
    write_trace_run(run, sequence=1, camera_name="overhead")
    write_trace_run(run, sequence=2, camera_name="left_wrist")
    output_path = tmp_path / "trace.mp4"

    count = trace_rerun.export_trace_video(run, output_path, fps=5.0, tile_width=32, tile_height=24)

    assert count == 2
    assert output_path.exists()
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    finally:
        capture.release()


def test_video_output_path_uses_run_name_for_parent_export(tmp_path):
    output_dir = tmp_path / "videos"
    trace_dir = tmp_path / "run-a"

    assert trace_rerun.video_output_path(output_dir, trace_dir, multiple_runs=True) == output_dir / "run-a.mp4"
