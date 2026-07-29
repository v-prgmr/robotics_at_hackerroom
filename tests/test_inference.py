import json
import sys
import time
import types

import numpy as np
import pytest
import torch

from bimanual_collection.hardware.bimanual_robot import BimanualFollowerState, DEFAULT_JOINT_NAMES
from bimanual_collection.hardware.cameras import CameraFrame, MatchedCameraFrame
from bimanual_collection.inference import (
    ActionChunkBuffer,
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
    TimedAction,
    bimanual_joint_vector,
    build_arg_parser,
    image_to_policy_tensor,
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


def test_split_bimanual_action_returns_left_and_right_joint_dicts():
    left, right = split_bimanual_action(np.arange(12, dtype=np.float32))

    assert left == {name: float(index) for index, name in enumerate(DEFAULT_JOINT_NAMES)}
    assert right == {name: float(index + 6) for index, name in enumerate(DEFAULT_JOINT_NAMES)}


def test_action_chunk_buffer_tracks_chunk_progress_and_replan_gates():
    buffer = ActionChunkBuffer()
    now = time.monotonic()

    chunk_id = buffer.publish_chunk(
        np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        source_timestamp_s=now,
    )
    assert chunk_id == 1
    assert not buffer.should_request_chunk(
        execution_horizon=2,
        replan_threshold=0,
        chunk_execution_mode=ChunkExecutionMode.RECEDING,
    )

    action = buffer.pop_action(now, max_age_s=1.0)
    assert action is not None
    np.testing.assert_array_equal(action, np.asarray([1, 2], dtype=np.float32))
    assert not buffer.should_request_chunk(
        execution_horizon=2,
        replan_threshold=0,
        chunk_execution_mode=ChunkExecutionMode.RECEDING,
    )

    popped = buffer.pop_timed_action(now, max_age_s=1.0)
    assert popped is not None
    np.testing.assert_array_equal(popped.action.action, np.asarray([3, 4], dtype=np.float32))
    assert popped.action.chunk_id == 1
    assert popped.action.action_index == 1
    assert popped.queue_remaining == 1
    assert buffer.should_request_chunk(
        execution_horizon=2,
        replan_threshold=0,
        chunk_execution_mode=ChunkExecutionMode.RECEDING,
    )

    chunk_id = buffer.publish_chunk(np.asarray([[7, 8]], dtype=np.float32), source_timestamp_s=now)
    assert chunk_id == 2
    action = buffer.pop_action(now, max_age_s=1.0)
    assert action is not None
    np.testing.assert_array_equal(action, np.asarray([7, 8], dtype=np.float32))

    buffer.publish_chunk(np.asarray([[7, 8]], dtype=np.float32), source_timestamp_s=now - 2.0)
    assert buffer.pop_action(now, max_age_s=1.0) is None


def test_action_chunk_buffer_can_disable_age_check():
    buffer = ActionChunkBuffer()
    now = time.monotonic()

    buffer.publish_chunk(np.asarray([[7, 8]], dtype=np.float32), source_timestamp_s=now - 100.0)

    action = buffer.pop_action(now, max_age_s=None)
    assert action is not None
    np.testing.assert_array_equal(action, np.asarray([7, 8], dtype=np.float32))


def test_action_chunk_buffer_full_mode_requests_only_when_empty():
    buffer = ActionChunkBuffer()
    now = time.monotonic()

    buffer.publish_chunk(np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32), source_timestamp_s=now)

    assert not buffer.should_request_chunk(
        execution_horizon=1,
        replan_threshold=10,
        chunk_execution_mode=ChunkExecutionMode.FULL,
    )
    assert buffer.pop_action(now, max_age_s=1.0) is not None
    assert not buffer.should_request_chunk(
        execution_horizon=1,
        replan_threshold=10,
        chunk_execution_mode=ChunkExecutionMode.FULL,
    )
    assert buffer.pop_action(now, max_age_s=1.0) is not None
    assert buffer.pop_action(now, max_age_s=1.0) is not None
    assert buffer.should_request_chunk(
        execution_horizon=1,
        replan_threshold=10,
        chunk_execution_mode=ChunkExecutionMode.FULL,
    )


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
        timed_action = ActionChunkBuffer()
        timed_action.publish_chunk(
            np.arange(12, dtype=np.float32)[None, :],
            source_timestamp_s=time.monotonic(),
        )
        popped = timed_action.pop_timed_action(time.monotonic(), max_age_s=1.0)
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
    assert events[2]["queue_remaining"] == 0
    assert events[2]["dry_run"] is True


def test_debug_trace_flag_parses_with_required_checkpoint():
    parser = build_arg_parser()

    args = parser.parse_args(["--checkpoint", "model", "--debug-trace-dir", "trace"])

    assert args.debug_trace_dir.name == "trace"
    assert args.execution_horizon == 10
    assert args.replan_threshold == 5
    assert args.chunk_execution_mode == ChunkExecutionMode.RECEDING
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
