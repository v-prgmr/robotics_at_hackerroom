import json
import sys
import threading
import time
import types
from typing import Any, cast

import numpy as np
import pytest
import torch

from bimanual_collection.hardware.bimanual_robot import (
    BimanualCommandResult,
    BimanualFollowerState,
    DEFAULT_JOINT_NAMES,
)
from bimanual_collection.hardware.cameras import CameraFrame, MatchedCameraFrame
from bimanual_collection.inference import (
    ActionChunkBuffer,
    AutonomousRolloutCapture,
    ChunkExecutionMode,
    DebugTraceWriter,
    DeploymentState,
    DeploymentHotkeys,
    LatestObservationBuffer,
    ObservationAdapter,
    ObservationSnapshot,
    RerunLiveVisualizer,
    RerunTelemetryConfig,
    StateTransitionLogger,
    StopFlag,
    TimedAction,
    bimanual_joint_vector,
    build_arg_parser,
    image_to_policy_tensor,
    run_control_loop,
    split_bimanual_action,
)


def _joints(offset=0.0):
    return {name: float(index + offset) for index, name in enumerate(DEFAULT_JOINT_NAMES)}


def _match(camera_name, value=100, stale=False):
    frame = CameraFrame(
        camera_name=camera_name,
        frame_index=1,
        image=np.full((4, 5, 3), value, dtype=np.uint8),
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
        stale=stale,
        missing=False,
        dropped_frames=0,
        disconnected=False,
    )


def test_bimanual_joint_vector_matches_lerobot_export_order():
    vector = bimanual_joint_vector(_joints(0), _joints(10))

    np.testing.assert_array_equal(
        vector,
        np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )


def test_image_to_policy_tensor_is_channel_first_float_unit_range():
    image = np.asarray([[[0, 127, 255]]], dtype=np.uint8)

    tensor = image_to_policy_tensor(image)

    assert tensor.shape == (3, 1, 1)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([0.0, 127.0 / 255.0, 1.0]))


def test_observation_adapter_builds_policy_schema():
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    snapshot = ObservationSnapshot(
        sequence=1,
        timestamp_s=1.0,
        follower_state=state,
        camera_matches={"left_wrist": _match("left_wrist")},
    )
    adapter = ObservationAdapter(
        task_description="Pick the teabag.",
        image_feature_keys=["observation.images.left_wrist"],
    )

    observation = adapter.build(snapshot)

    assert observation["task"] == "Pick the teabag."
    assert observation["observation.state"].shape == (12,)
    assert observation["observation.images.left_wrist"].shape == (3, 4, 5)


def test_observation_adapter_rejects_stale_camera():
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    snapshot = ObservationSnapshot(
        sequence=1,
        timestamp_s=1.0,
        follower_state=state,
        camera_matches={"left_wrist": _match("left_wrist", stale=True)},
    )
    adapter = ObservationAdapter(
        task_description="Pick the teabag.",
        image_feature_keys=["observation.images.left_wrist"],
    )

    with pytest.raises(ValueError, match="Invalid camera frame"):
        adapter.build(snapshot)


def test_latest_observation_buffer_clear_drops_stale_observation():
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    buffer = LatestObservationBuffer()

    sequence = buffer.publish(
        timestamp_s=1.0,
        follower_state=state,
        camera_matches={"left_wrist": _match("left_wrist")},
    )
    assert buffer.wait_for_new(0, timeout_s=0.001) is not None

    buffer.clear()

    assert buffer.wait_for_new(sequence, timeout_s=0.001) is None


def test_latest_observation_buffer_selects_policy_period_history():
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    buffer = LatestObservationBuffer()
    snapshots = []
    for timestamp in (1.0, 1.02, 1.04, 1.06):
        sequence = buffer.publish(timestamp_s=timestamp, follower_state=state, camera_matches={})
        snapshot = buffer.wait_for_new(sequence - 1, timeout_s=0.001)
        assert snapshot is not None
        snapshots.append(snapshot)

    history = buffer.history_ending_at(snapshots[-1], steps=2, period_s=0.06)

    assert [item.timestamp_s for item in history] == [1.0, 1.06]


def test_split_bimanual_action_returns_left_and_right_joint_dicts():
    left, right = split_bimanual_action(np.arange(12, dtype=np.float32))

    assert left == {name: float(index) for index, name in enumerate(DEFAULT_JOINT_NAMES)}
    assert right == {name: float(index + 6) for index, name in enumerate(DEFAULT_JOINT_NAMES)}


def test_action_chunk_buffer_interpolates_by_elapsed_time_and_replans_by_knots():
    buffer = ActionChunkBuffer(policy_action_hz=10.0, bridge_duration_s=0.0)
    token = buffer.begin_request(source_timestamp_s=10.0, hold_action=np.asarray([0, 0], dtype=np.float32))
    assert token is not None
    assert buffer.publish_chunk(token, np.asarray([[0, 0], [10, 20], [20, 40]], dtype=np.float32)) == 1

    sampled = buffer.sample_timed_action(20.0, np.asarray([0, 0], dtype=np.float32), max_age_s=None)
    assert sampled is not None
    np.testing.assert_array_equal(sampled.action.action, np.asarray([0, 0], dtype=np.float32))

    sampled = buffer.sample_timed_action(20.05, np.asarray([0, 0], dtype=np.float32), max_age_s=None)
    assert sampled is not None
    np.testing.assert_allclose(sampled.action.action, np.asarray([5, 10], dtype=np.float32), atol=1e-4)
    assert sampled.action.action_index == 0
    assert sampled.action.upper_action_index == 1
    assert sampled.action.interpolation_alpha == pytest.approx(0.5)
    assert not buffer.should_request_chunk(2, 0, ChunkExecutionMode.RECEDING, now_s=20.05)
    assert buffer.should_request_chunk(2, 0, ChunkExecutionMode.RECEDING, now_s=20.1)


def test_action_chunk_buffer_latches_request_pose_and_rejects_late_response():
    buffer = ActionChunkBuffer(policy_action_hz=10.0, bridge_duration_s=0.0)
    hold = np.asarray([1, 2], dtype=np.float32)
    token = buffer.begin_request(source_timestamp_s=1.0, hold_action=hold)
    assert token is not None
    hold[:] = 99

    first = buffer.sample_timed_action(1.1, np.asarray([5, 6], dtype=np.float32), max_age_s=1.0)
    second = buffer.sample_timed_action(1.2, np.asarray([7, 8], dtype=np.float32), max_age_s=1.0)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.action.action, np.asarray([1, 2], dtype=np.float32))
    np.testing.assert_array_equal(second.action.action, np.asarray([1, 2], dtype=np.float32))
    assert first.action.phase == "request_hold"

    buffer.clear()
    assert buffer.publish_chunk(token, np.asarray([[3, 4]], dtype=np.float32)) is None
    assert buffer.sample_timed_action(1.3, np.asarray([1, 2], dtype=np.float32), max_age_s=1.0) is None


def test_action_chunk_buffer_full_mode_requests_after_last_knot_interval():
    buffer = ActionChunkBuffer(policy_action_hz=10.0, bridge_duration_s=0.0)
    token = buffer.begin_request(source_timestamp_s=1.0, hold_action=np.asarray([0, 0], dtype=np.float32))
    assert token is not None
    buffer.publish_chunk(token, np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    assert buffer.sample_timed_action(2.0, np.asarray([0, 0], dtype=np.float32), max_age_s=None) is not None
    assert not buffer.should_request_chunk(1, 10, ChunkExecutionMode.FULL, now_s=2.29)
    assert buffer.should_request_chunk(1, 10, ChunkExecutionMode.FULL, now_s=2.3)


def test_action_chunk_buffer_bridges_from_response_state_before_trajectory():
    buffer = ActionChunkBuffer(policy_action_hz=10.0, bridge_duration_s=0.1)
    token = buffer.begin_request(source_timestamp_s=1.0, hold_action=np.asarray([0], dtype=np.float32))
    assert token is not None
    buffer.publish_chunk(token, np.asarray([[10], [20]], dtype=np.float32))

    start = buffer.sample_timed_action(2.0, np.asarray([4], dtype=np.float32), max_age_s=None)
    middle = buffer.sample_timed_action(2.05, np.asarray([5], dtype=np.float32), max_age_s=None)
    action_zero = buffer.sample_timed_action(2.1, np.asarray([6], dtype=np.float32), max_age_s=None)
    assert start is not None and middle is not None and action_zero is not None
    np.testing.assert_allclose(start.action.action, [4])
    np.testing.assert_allclose(middle.action.action, [7], atol=1e-4)
    np.testing.assert_allclose(action_zero.action.action, [10], atol=1e-4)
    assert start.action.phase == "bridge"
    assert action_zero.action.phase == "trajectory"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy_action_hz": float("nan")}, "policy_action_hz"),
        ({"policy_action_hz": float("inf")}, "policy_action_hz"),
        ({"bridge_duration_s": float("nan")}, "bridge_duration_s"),
        ({"bridge_duration_s": float("inf")}, "bridge_duration_s"),
    ],
)
def test_action_chunk_buffer_rejects_nonfinite_timing(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ActionChunkBuffer(**kwargs)


def test_state_transition_logger_prints_only_changes(capsys):
    logger = StateTransitionLogger()

    logger.set(DeploymentState.IDLE, "press the arm key")
    logger.set(DeploymentState.IDLE, "ignored")
    logger.set(DeploymentState.RUNNING, "policy commands active")

    assert capsys.readouterr().out.splitlines() == [
        "State -> IDLE: press the arm key",
        "State -> RUNNING: policy commands active",
    ]


def test_deployment_hotkeys_consume_arm_and_reset_edges():
    hotkeys = DeploymentHotkeys()

    assert hotkeys.consume_arm_toggle() is False
    assert hotkeys.consume_reset() is False
    assert hotkeys.consume_home() is False

    hotkeys.request_arm_toggle()
    hotkeys.request_reset()
    hotkeys.request_home()

    assert hotkeys.consume_arm_toggle() is True
    assert hotkeys.consume_reset() is True
    assert hotkeys.consume_home() is True
    assert hotkeys.consume_arm_toggle() is False
    assert hotkeys.consume_reset() is False
    assert hotkeys.consume_home() is False


def test_deployment_hotkeys_keep_first_pending_rollout_classification():
    hotkeys = DeploymentHotkeys()

    hotkeys.request_classification("failure")
    hotkeys.request_classification("success")

    assert hotkeys.consume_classification() == "failure"
    assert hotkeys.consume_classification() is None


def test_rollout_capture_selects_policy_rate_ticks_from_60_hz_controller(tmp_path):
    capture = AutonomousRolloutCapture(
        recorder=cast(Any, object()),
        success_output_dir=tmp_path / "successes",
        failure_output_dir=tmp_path / "failures",
        policy_checkpoint=None,
        capture_hz=16.57,
    )

    selected = [tick / 60.0 for tick in range(600) if capture.should_record(tick / 60.0)]

    assert len(selected) == 166
    assert (len(selected) - 1) / (selected[-1] - selected[0]) == pytest.approx(16.57, rel=0.01)


def test_rollout_capture_does_not_emit_catch_up_bursts(tmp_path):
    capture = AutonomousRolloutCapture(
        recorder=cast(Any, object()),
        success_output_dir=tmp_path / "successes",
        failure_output_dir=tmp_path / "failures",
        policy_checkpoint=None,
        capture_hz=16.57,
    )

    assert capture.should_record(0.0) is True
    assert capture.should_record(0.01) is False
    assert capture.should_record(1.0) is True
    assert capture.should_record(1.0) is False


def test_rollout_classification_records_current_hold_tick_before_disarming():
    stop_requested = threading.Event()
    hotkeys = DeploymentHotkeys()
    hotkeys.request_arm_toggle()
    hotkeys.request_classification("success")
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)

    class FakeRecorder:
        is_recording = False
        episode_id = None
        sample_count = 0
        samples = []

        def start_deferred(self, **_kwargs):
            self.is_recording = True
            self.episode_id = "capture-test"

        def add_sample(self, sample):
            self.samples.append(sample)
            self.sample_count += 1

        def discard(self):
            self.is_recording = False

    class FakeCapture:
        recorder = FakeRecorder()

        def start(self):
            self.recorder.start_deferred()

        def classify(self, classification):
            assert classification == "success"
            self.recorder.is_recording = False
            stop_requested.set()
            return "successes/episode-000001"

        def discard(self):
            self.recorder.discard()

    class FakeRobot:
        is_connected = False

        def read_state(self):
            return state

        def hold_position(self):
            return BimanualCommandResult(_joints(0.5), _joints(10.5), 0.0, 0.0)

    worker = types.SimpleNamespace(error=None)
    cameras = types.SimpleNamespace(match=lambda _timestamp: types.SimpleNamespace(matches={}))
    armed = threading.Event()
    reset_requested = threading.Event()
    capture = FakeCapture()

    run_control_loop(
        robot=cast(Any, FakeRobot()),
        cameras=cast(Any, cameras),
        policy_worker=cast(Any, worker),
        observation_buffer=LatestObservationBuffer(),
        action_buffer=ActionChunkBuffer(),
        armed=armed,
        reset_requested=reset_requested,
        policy_stop_requested=stop_requested,
        hotkeys=hotkeys,
        robot_fps=60,
        image_feature_keys=[],
        max_action_age_s=1.0,
        max_action_delta=None,
        home_callback=lambda: None,
        dry_run=False,
        flag=StopFlag(),
        rollout_capture=cast(Any, capture),
    )

    assert len(capture.recorder.samples) == 1
    sample = capture.recorder.samples[0]
    assert sample.left_commanded_action == _joints(0.5)
    assert sample.right_commanded_action == _joints(10.5)
    assert sample.metadata["action_source"] == "hold"
    assert not armed.is_set()
    assert reset_requested.is_set()


def test_checkpoint_flag_is_required():
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_debug_trace_writes_observation_chunk_and_action(tmp_path):
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    snapshot = ObservationSnapshot(
        sequence=1,
        timestamp_s=1.0,
        follower_state=state,
        camera_matches={"left_wrist": _match("left_wrist")},
    )
    trace = DebugTraceWriter(tmp_path, metadata={"task_description": "Pick."})
    try:
        trace.log_policy_observation(
            snapshot,
            image_feature_keys=["observation.images.left_wrist"],
            joint_names=DEFAULT_JOINT_NAMES,
        )
        trace.log_policy_chunk(
            snapshot,
            chunk_id=1,
            normalized_actions=np.ones((2, 12), dtype=np.float32),
            postprocessed_actions=np.arange(24, dtype=np.float32).reshape(2, 12),
            inference_duration_s=0.25,
        )
        timed_action = ActionChunkBuffer(bridge_duration_s=0.0)
        now = time.monotonic()
        token = timed_action.begin_request(source_timestamp_s=now, hold_action=np.zeros(12, dtype=np.float32))
        assert token is not None
        timed_action.publish_chunk(token, np.arange(12, dtype=np.float32)[None, :])
        popped = timed_action.sample_timed_action(now, np.zeros(12, dtype=np.float32), max_age_s=1.0)
        assert popped is not None
        left, right = split_bimanual_action(popped.action.action)
        trace.log_robot_action(
            action=popped.action.action,
            left_action=left,
            right_action=right,
            timed_action=popped.action,
            queue_remaining=popped.queue_remaining,
            dry_run=True,
        )
    finally:
        trace.close()

    assert (trace.run_dir / "metadata.json").exists()
    assert (trace.run_dir / "images/obs_000001_left_wrist.png").exists()
    assert (trace.run_dir / "action_chunks/chunk_000001_obs_000001_normalized.npy").exists()
    assert (trace.run_dir / "action_chunks/chunk_000001_obs_000001_postprocessed.npy").exists()

    events = [json.loads(line) for line in (trace.run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "policy_observation",
        "policy_chunk",
        "robot_action",
    ]
    assert events[0]["cameras"][0]["image_path"] == "images/obs_000001_left_wrist.png"
    assert events[1]["chunk_id"] == 1
    assert events[1]["postprocessed_action_shape"] == [2, 12]
    assert events[2]["chunk_id"] == 1
    assert events[2]["action_index"] == 0
    assert events[2]["upper_action_index"] == 0
    assert events[2]["interpolation_alpha"] == 0.0
    assert events[2]["phase"] == "trajectory"
    assert events[2]["queue_remaining"] == 0
    assert events[2]["dry_run"] is True


def test_debug_trace_flag_parses_with_required_checkpoint():
    parser = build_arg_parser()

    args = parser.parse_args(["--checkpoint", "model", "--debug-trace-dir", "trace"])

    assert args.debug_trace_dir.name == "trace"
    assert args.execution_horizon == 10
    assert args.replan_threshold == 5
    assert args.chunk_execution_mode == ChunkExecutionMode.FULL
    assert args.policy_action_hz is None
    assert args.action_bridge_duration_s is None
    assert args.disable_action_age_check is False
    assert args.reset_key == "r"
    assert args.home_key == "h"
    assert args.rerun_live is False
    assert args.rerun_camera_fps == 10.0
    assert args.rerun_max_queue == 512


def test_rerun_flags_parse_with_required_checkpoint():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--checkpoint",
            "model",
            "--rerun-live",
            "--rerun-save",
            "run.rrd",
            "--rerun-camera-fps",
            "5",
            "--rerun-max-queue",
            "12",
        ]
    )

    assert args.rerun_live is True
    assert args.rerun_save.name == "run.rrd"
    assert args.rerun_camera_fps == 5
    assert args.rerun_max_queue == 12


def test_rerun_live_visualizer_logs_core_runtime_telemetry(monkeypatch, tmp_path):
    calls = []

    def archetype(name):
        def build(*args, **kwargs):
            return (name, args, kwargs)

        return build

    fake_rr = types.SimpleNamespace(
        Image=archetype("Image"),
        Scalars=archetype("Scalars"),
        LineStrips2D=archetype("LineStrips2D"),
        TextLog=archetype("TextLog"),
        init=lambda *args, **kwargs: calls.append(("init", args, kwargs)),
        connect_grpc=lambda *args, **kwargs: calls.append(("connect_grpc", args, kwargs)),
        save=lambda *args, **kwargs: calls.append(("save", args, kwargs)),
        set_time_seconds=lambda *args, **kwargs: calls.append(("set_time_seconds", args, kwargs)),
        set_time_sequence=lambda *args, **kwargs: calls.append(("set_time_sequence", args, kwargs)),
        log=lambda *args, **kwargs: calls.append(("log", args, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    state = BimanualFollowerState(_joints(0), _joints(10), 0.0, 0.0)
    snapshot = ObservationSnapshot(
        sequence=1,
        timestamp_s=1.0,
        follower_state=state,
        camera_matches={"left_wrist": _match("left_wrist")},
    )
    visualizer = RerunLiveVisualizer(
        RerunTelemetryConfig(
            spawn=False,
            connect_grpc_url="rerun+http://127.0.0.1:9876/proxy",
            save_path=tmp_path / "run.rrd",
            camera_fps=1000,
        )
    )

    visualizer.start()
    visualizer.log_observation(
        snapshot,
        image_feature_keys=["observation.images.left_wrist"],
        joint_names=DEFAULT_JOINT_NAMES,
    )
    visualizer.log_policy_chunk(
        snapshot,
        chunk_id=1,
        postprocessed_actions=np.arange(24, dtype=np.float32).reshape(2, 12),
        inference_duration_s=0.25,
    )
    visualizer.log_robot_action(
        action=np.arange(12, dtype=np.float32),
        left_action=_joints(0),
        right_action=_joints(10),
        timed_action=TimedAction(np.arange(12, dtype=np.float32), 0.5, 0.75, 1, 0),
        queue_remaining=3,
        dry_run=True,
    )
    visualizer.log_hold(reason="missing or stale policy action", armed=True)
    visualizer.log_state_transition(state=DeploymentState.RUNNING, reason="policy commands active")
    visualizer.close()

    log_paths = [call[1][0] for call in calls if call[0] == "log"]
    assert any(call[0] == "init" for call in calls)
    assert any(call[0] == "connect_grpc" for call in calls)
    assert any(call[0] == "save" for call in calls)
    assert "cameras/left_wrist/image" in log_paths
    assert "joints/left_follower/shoulder_pan_pos" in log_paths
    assert "policy_chunk/left/shoulder_pan_pos" in log_paths
    assert "commands/right_follower/gripper_pos" in log_paths
    assert "events/holds" in log_paths
    assert "events/state" in log_paths
    assert visualizer.dropped_events == 0


def test_disable_action_age_check_allows_nonpositive_max_age():
    parser = build_arg_parser()

    args = parser.parse_args(
        ["--checkpoint", "model", "--disable-action-age-check", "--max-action-age-s", "0"]
    )

    assert args.disable_action_age_check is True
    assert args.max_action_age_s == 0
