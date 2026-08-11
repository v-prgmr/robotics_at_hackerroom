import json
import time

import numpy as np
import pandas as pd

from bimanual_collection.hardware.cameras import CameraFrame, MatchedCameraFrame
from bimanual_collection.recording.episode import TimestepSample
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig


def make_sample(episode_id: str, index: int) -> TimestepSample:
    timestamp = 100.0 + index / 60.0
    frame = CameraFrame(
        camera_name="overhead",
        frame_index=index,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        camera_timestamp_s=timestamp,
        host_timestamp_s=timestamp,
    )
    match = MatchedCameraFrame(
        camera_name="overhead",
        frame=frame,
        frame_index=index,
        camera_timestamp_s=timestamp,
        host_timestamp_s=timestamp,
        hardware_timestamp_s=None,
        age_s=0.0,
        stale=False,
        missing=False,
        dropped_frames=0,
        disconnected=False,
    )
    joints = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 1.0,
        "elbow_flex.pos": 2.0,
        "wrist_flex.pos": 3.0,
        "wrist_roll.pos": 4.0,
        "gripper.pos": 5.0,
    }
    return TimestepSample(
        episode_id=episode_id,
        timestep_index=index,
        monotonic_timestamp_s=time.monotonic() + index / 60.0,
        wall_timestamp_s=time.time(),
        left_leader_joints=joints,
        right_leader_joints=joints,
        left_follower_joints=joints,
        right_follower_joints=joints,
        left_gripper_state=5.0,
        right_gripper_state=5.0,
        left_commanded_action=joints,
        right_commanded_action=joints,
        camera_matches={"overhead": match},
        measured_control_hz=60.0,
        loop_duration_s=0.005,
    )


def test_episode_recorder_saves_atomically(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, task_description="test", camera_fps=30))
    episode_id = recorder.start(episode_id="episode-test")
    recorder.add_sample(make_sample(episode_id, 0))
    recorder.add_sample(make_sample(episode_id, 1))
    path = recorder.stop_and_save(success=True, operator_notes="ok")

    assert path == tmp_path / "episode-test"
    assert path.exists()
    assert not list(tmp_path.glob(".episode-test.tmp-*"))
    assert (path / "timesteps.parquet").exists()
    assert (path / "camera_index.parquet").exists()
    assert (path / "control_events.parquet").exists()
    assert (path / "videos" / "overhead.mp4").exists()
    df = pd.read_parquet(path / "timesteps.parquet")
    assert len(df) == 2
    assert df["overhead_video_frame_index"].tolist() == [0, 1]


def test_episode_recorder_discards_temp_directory(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, task_description="test", camera_fps=30))
    episode_id = recorder.start(episode_id="episode-discard")
    recorder.add_sample(make_sample(episode_id, 0))
    recorder.discard()

    assert not (tmp_path / "episode-discard").exists()
    assert not list(tmp_path.glob(".episode-discard.tmp-*"))


def test_episode_recorder_exposes_current_recording_status(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, task_description="test", camera_fps=30))
    episode_id = recorder.start(episode_id="episode-status")

    assert recorder.is_recording
    assert recorder.episode_id == "episode-status"
    assert recorder.current_final_dir == tmp_path / "episode-status"
    assert recorder.sample_count == 0

    recorder.add_sample(make_sample(episode_id, 0))
    assert recorder.sample_count == 1
    assert recorder.elapsed_s >= 0.0

    recorder.discard()
    assert not recorder.is_recording
    assert recorder.episode_id is None
    assert recorder.sample_count == 0


def test_episode_recorder_assigns_sequential_episode_numbers(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, task_description="test", camera_fps=30))

    first_id = recorder.start()
    assert first_id == "episode-000001"
    assert recorder.episode_number == 1
    assert recorder.episode_label == "Episode 1"
    recorder.stop_and_save()

    second_id = recorder.start()
    assert second_id == "episode-000002"
    assert recorder.episode_number == 2
    assert recorder.episode_label == "Episode 2"
    recorder.discard()


def test_episode_recorder_can_start_from_configured_number(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, episode_start_number=51))

    episode_id = recorder.start()

    assert episode_id == "episode-000051"
    assert recorder.episode_number == 51
    assert recorder.episode_label == "Episode 51"
    recorder.discard()


def test_episode_recorder_skips_existing_configured_number(tmp_path):
    (tmp_path / "episode-000051").mkdir()
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, episode_start_number=51))

    episode_id = recorder.start()

    assert episode_id == "episode-000052"
    assert recorder.episode_number == 52
    recorder.discard()


def test_episode_recorder_saves_extra_metadata(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path, task_description="test", camera_fps=30))

    episode_id = recorder.start(episode_id="episode-extra", extra={"hil_protocol": "bounded"})
    recorder.add_sample(make_sample(episode_id, 0))
    path = recorder.stop_and_save()

    metadata = json.loads((path / "episode_metadata.json").read_text(encoding="utf-8"))
    assert metadata["extra"]["hil_protocol"] == "bounded"
    assert metadata["hil_protocol"] == "bounded"


def test_deferred_episode_numbering_is_independent_per_destination(tmp_path):
    staging = tmp_path / "staging"
    successes = tmp_path / "successes"
    failures = tmp_path / "failures"
    recorder = EpisodeRecorder(RecorderConfig(output_dir=staging, task_description="test", camera_fps=30))

    provisional_id = recorder.start_deferred(extra={"episode_type": "autonomous_rollout"})
    recorder.add_sample(make_sample(provisional_id, 0))
    success_path = recorder.stop_and_save_to(
        successes,
        success=True,
        extra_metadata={"terminal_success": True, "termination_reason": "operator_success"},
    )

    provisional_id = recorder.start_deferred(extra={"episode_type": "autonomous_rollout"})
    recorder.add_sample(make_sample(provisional_id, 0))
    failure_path = recorder.stop_and_save_to(
        failures,
        success=False,
        extra_metadata={"terminal_success": False, "termination_reason": "operator_failure"},
    )

    assert success_path == successes / "episode-000001"
    assert failure_path == failures / "episode-000001"
    success_rows = pd.read_parquet(success_path / "timesteps.parquet")
    failure_rows = pd.read_parquet(failure_path / "timesteps.parquet")
    assert success_rows["episode_id"].tolist() == ["episode-000001"]
    assert failure_rows["episode_id"].tolist() == ["episode-000001"]
    success_metadata = json.loads((success_path / "episode_metadata.json").read_text(encoding="utf-8"))
    failure_metadata = json.loads((failure_path / "episode_metadata.json").read_text(encoding="utf-8"))
    assert success_metadata["terminal_success"] is True
    assert success_metadata["termination_reason"] == "operator_success"
    assert failure_metadata["terminal_success"] is False
    assert failure_metadata["termination_reason"] == "operator_failure"
    assert not list(staging.glob(".*.tmp-*"))
