from io import BytesIO

import numpy as np
import pytest

from bimanual_collection.hardware.bimanual_robot import BimanualFollowerState, DEFAULT_JOINT_NAMES
from bimanual_collection.hardware.cameras import CameraFrame, MatchedCameraFrame
from bimanual_collection.inference import ObservationSnapshot, build_arg_parser
from bimanual_collection.maniflow_inference import (
    _normalize_server_url,
    build_runtime_and_adapter,
    configure_parser,
)
from bimanual_collection.maniflow_remote import ManiFlowRemoteObservationAdapter, ManiFlowRemotePolicyRuntime


def _joints(offset=0.0):
    return {name: float(index + offset) for index, name in enumerate(DEFAULT_JOINT_NAMES)}


def _match(camera_name):
    frame = CameraFrame(
        camera_name=camera_name,
        frame_index=1,
        image=np.full((4, 5, 3), 100, dtype=np.uint8),
        camera_timestamp_s=1.0,
        host_timestamp_s=1.0,
    )
    return MatchedCameraFrame(
        camera_name=camera_name,
        frame=frame,
        frame_index=1,
        camera_timestamp_s=1.0,
        host_timestamp_s=1.0,
        hardware_timestamp_s=None,
        age_s=0.0,
        stale=False,
        missing=False,
        dropped_frames=0,
        disconnected=False,
    )


def test_maniflow_remote_observation_adapter_builds_server_schema():
    snapshot = ObservationSnapshot(
        sequence=1,
        timestamp_s=1.0,
        follower_state=BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0),
        camera_matches={camera: _match(camera) for camera in ("overhead", "left_wrist", "right_wrist")},
    )
    adapter = ManiFlowRemoteObservationAdapter(task_description="Pick one teabag.")

    observation = adapter.build(snapshot)

    assert observation["task_name"] == "Pick one teabag."
    np.testing.assert_array_equal(
        observation["agent_pos"],
        np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )
    assert sorted(observation["images"]) == ["left_wrist", "overhead", "right_wrist"]
    assert observation["images"]["overhead"].shape == (4, 5, 3)


def test_maniflow_remote_observation_adapter_builds_time_history():
    snapshots = [
        ObservationSnapshot(
            sequence=index,
            timestamp_s=float(index),
            follower_state=BimanualFollowerState(_joints(index), _joints(10 + index), 0.0, 0.0),
            camera_matches={camera: _match(camera) for camera in ("overhead", "left_wrist", "right_wrist")},
        )
        for index in (1, 2)
    ]
    adapter = ManiFlowRemoteObservationAdapter(task_description="Pick one teabag.")

    observation = adapter.build_history(snapshots)

    assert observation["agent_pos"].shape == (2, 12)
    assert observation["images"]["overhead"].shape == (2, 4, 5, 3)


def test_maniflow_remote_policy_runtime_posts_npz_and_reads_action_chunk(monkeypatch):
    runtime = ManiFlowRemotePolicyRuntime("http://127.0.0.1:8765")
    observation = {
        "agent_pos": np.arange(12, dtype=np.float32),
        "task_name": "Pick one teabag.",
        "images": {
            "overhead": np.zeros((4, 5, 3), dtype=np.uint8),
            "left_wrist": np.ones((4, 5, 3), dtype=np.uint8),
            "right_wrist": np.full((4, 5, 3), 2, dtype=np.uint8),
        },
    }

    def fake_post(path, body):
        assert path == "/predict"
        with np.load(BytesIO(body), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["agent_pos"], np.arange(12, dtype=np.float32))
            assert str(arrays["task_name"].item()) == "Pick one teabag."
            assert arrays["image_left_wrist"].shape == (4, 5, 3)
        buffer = BytesIO()
        np.savez_compressed(buffer, actions=np.arange(24, dtype=np.float32).reshape(2, 12))
        return buffer.getvalue()

    monkeypatch.setattr(runtime, "_post_bytes", fake_post)

    normalized_actions, actions = runtime.predict_action_chunk_with_debug(observation)

    assert normalized_actions is None
    np.testing.assert_array_equal(actions, np.arange(24, dtype=np.float32).reshape(2, 12))
    assert runtime.latest_progress is None


def test_maniflow_remote_reads_optional_progress_without_changing_action_api(monkeypatch):
    runtime = ManiFlowRemotePolicyRuntime("http://127.0.0.1:8765")
    observation = {
        "agent_pos": np.arange(12, dtype=np.float32),
        "task_name": "Pick one teabag.",
        "images": {
            camera: np.zeros((4, 5, 3), dtype=np.uint8)
            for camera in ("overhead", "left_wrist", "right_wrist")
        },
    }

    def fake_post(_path, _body):
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            actions=np.arange(12, dtype=np.float32)[None],
            progress=np.asarray([[0.625]], dtype=np.float32),
        )
        return buffer.getvalue()

    monkeypatch.setattr(runtime, "_post_bytes", fake_post)

    actions = runtime.predict_action_chunk(observation)

    assert actions.shape == (1, 12)
    assert runtime.latest_progress == pytest.approx(0.625)


def test_normalize_maniflow_server_url_adds_http_scheme():
    assert _normalize_server_url("127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert _normalize_server_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765"


def test_maniflow_remote_runtime_reads_server_settings_from_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "maniflow_server: 192.0.2.1:9999\nmaniflow_timeout_s: 3.5\nmaniflow_cameras: [overhead]\n"
    )
    parser = build_arg_parser(checkpoint_required=False)
    configure_parser(parser)
    args = parser.parse_args(["--config", str(config)])

    runtime, adapter = build_runtime_and_adapter(args, "Pick one teabag.")

    assert runtime.server_url == "http://192.0.2.1:9999"
    assert runtime.timeout_s == 3.5
    assert adapter.cameras == ("overhead",)


def test_maniflow_rollout_capture_flags_parse_without_rerun():
    parser = build_arg_parser(checkpoint_required=False)
    configure_parser(parser)

    args = parser.parse_args(
        [
            "--capture-dataset",
            "--failure-output-dir",
            "failures",
            "--success-output-dir",
            "successes",
        ]
    )

    assert args.capture_dataset is True
    assert args.failure_output_dir.name == "failures"
    assert args.success_output_dir.name == "successes"
    assert args.failure_key == "f"
    assert args.success_key == "s"
    assert args.rerun_live is False
    assert args.rerun_save is None
