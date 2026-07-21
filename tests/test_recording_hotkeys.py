from argparse import Namespace

from bimanual_collection.bimanual_teleop import RecordingHotkeys, apply_recording_request, build_recording_control_config
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig


def test_recording_hotkeys_are_consumed_once():
    hotkeys = RecordingHotkeys()

    hotkeys.request_start_save()
    assert hotkeys.consume() == (True, False)
    assert hotkeys.consume() == (False, False)

    hotkeys.request_cancel()
    assert hotkeys.consume() == (False, True)
    assert hotkeys.consume() == (False, False)


def test_recording_hotkeys_can_report_both_edges():
    hotkeys = RecordingHotkeys()

    hotkeys.request_start_save()
    hotkeys.request_cancel()

    assert hotkeys.consume() == (True, True)


def test_start_save_request_saves_and_stays_idle_until_next_request(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path))

    first_episode_id = apply_recording_request(recorder, start_save_requested=True, cancel_requested=False)
    assert first_episode_id is not None
    assert recorder.is_recording

    saved_episode_id = apply_recording_request(recorder, start_save_requested=True, cancel_requested=False)
    assert saved_episode_id is None
    assert not recorder.is_recording
    assert len([path for path in tmp_path.iterdir() if path.name.startswith("episode-")]) == 1

    idle_episode_id = apply_recording_request(recorder, start_save_requested=False, cancel_requested=False)
    assert idle_episode_id is None
    assert not recorder.is_recording
    assert len([path for path in tmp_path.iterdir() if path.name.startswith("episode-")]) == 1

    second_episode_id = apply_recording_request(recorder, start_save_requested=True, cancel_requested=False)
    assert second_episode_id is not None
    assert second_episode_id != first_episode_id
    assert recorder.is_recording


def test_recording_control_can_disable_keyboard_cancel():
    args = Namespace(
        recording_manual_start=None,
        recording_status_interval_s=None,
        record_start_save_key=None,
        record_cancel_key=None,
    )

    config = {"recording_control": {"manual_start": True, "start_save_key": "r", "cancel_key": None}}

    recording_control = build_recording_control_config(args, config)
    assert recording_control.start_save_key == "r"
    assert recording_control.cancel_key is None
