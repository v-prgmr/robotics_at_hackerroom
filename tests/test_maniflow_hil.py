import argparse
import threading
import time
from typing import Any, cast

import numpy as np
import pytest

from bimanual_collection.hardware.bimanual_robot import BimanualFollowerState, DEFAULT_JOINT_NAMES
from bimanual_collection.inference import ActionChunkBuffer, LatestObservationBuffer, PeriodicCaptureGate, PolicyWorker
from bimanual_collection.maniflow_hil import (
    HILControlConfig,
    HILControlSnapshot,
    HILControls,
    HILGenerationGate,
    HILPhase,
    HILProtocolConfig,
    HILRolloutState,
    build_hil_protocol_config,
    build_hil_control_config,
    correction_episode_metadata,
    next_phase_after_saved_correction,
    next_hil_phase,
    _finish_correction,
)
from bimanual_collection.recording.recorder import EpisodeRecorder, RecorderConfig


def snapshot(
    *,
    pause: bool = False,
    correction: bool = False,
    arm: bool = False,
    reset: bool = False,
    clutch: bool = False,
) -> HILControlSnapshot:
    return HILControlSnapshot(
        pause_resume_edge=pause,
        correction_edge=correction,
        arm_edge=arm,
        reset_edge=reset,
        clutch_active=clutch,
        failed=False,
        error=None,
    )


def test_hil_state_machine_pauses_and_resumes_autonomous():
    assert next_hil_phase(HILPhase.READY_TO_ARM, snapshot(arm=True)) == HILPhase.AUTONOMOUS

    phase = next_hil_phase(HILPhase.AUTONOMOUS, snapshot(pause=True))
    assert phase == HILPhase.PAUSED

    phase = next_hil_phase(phase, snapshot(pause=True))
    assert phase == HILPhase.AUTONOMOUS


def test_hil_state_machine_records_corrections_only_from_paused():
    assert next_hil_phase(HILPhase.AUTONOMOUS, snapshot(correction=True)) == HILPhase.AUTONOMOUS

    phase = next_hil_phase(HILPhase.PAUSED, snapshot(correction=True))
    assert phase == HILPhase.PRE_CORRECTION

    phase = next_hil_phase(phase, snapshot(correction=True))
    assert phase == HILPhase.CORRECTING

    phase = next_hil_phase(phase, snapshot(correction=True))
    assert phase == HILPhase.PAUSED


def test_hil_state_machine_ignores_pause_while_correcting():
    assert next_hil_phase(HILPhase.CORRECTING, snapshot(pause=True)) == HILPhase.CORRECTING


def test_hil_correction_capture_matches_expert_rate_from_60_hz_ticks():
    gate = PeriodicCaptureGate(16.57)

    selected = [tick / 60.0 for tick in range(600) if gate.should_capture(tick / 60.0)]

    assert len(selected) == 166
    assert (len(selected) - 1) / (selected[-1] - selected[0]) == pytest.approx(16.57, rel=0.01)


def test_hil_correction_capture_gate_resets_for_each_correction():
    gate = PeriodicCaptureGate(16.57)

    assert gate.should_capture(10.0) is True
    assert gate.should_capture(10.01) is False
    gate.reset()
    assert gate.should_capture(10.01) is True


def test_hil_state_machine_can_cancel_pre_correction_setup():
    phase = next_hil_phase(HILPhase.PAUSED, snapshot(correction=True))
    assert phase == HILPhase.PRE_CORRECTION

    phase = next_hil_phase(phase, snapshot(pause=True))
    assert phase == HILPhase.PAUSED


def test_build_hil_control_config_reads_config_defaults():
    args = argparse.Namespace(
        hil_pedal_backend=None,
        hil_pedal_device=None,
        hil_pedal_debounce_s=None,
        hil_pause_code=None,
        hil_correction_code=None,
        hil_clutch_code=None,
        hil_arm_code=None,
        hil_reset_code=None,
        hil_protocol=None,
        hil_max_interventions_per_rollout=None,
    )
    cfg = build_hil_control_config(
        args,
        {
            "maniflow_hil": {
                "pedal_backend": "keyboard",
                "pedal_device": "/dev/input/test",
                "pedal_debounce_s": 0.2,
                "pause_code": "p",
                "correction_code": "c",
                "clutch_code": "space",
            }
        },
    )

    assert cfg.backend == "keyboard"
    assert cfg.device == "/dev/input/test"
    assert cfg.debounce_s == 0.2
    assert cfg.pause_resume == "p"
    assert cfg.correction == "c"
    assert cfg.clutch == "space"
    assert cfg.arm == "space"
    assert cfg.reset == "r"


def test_build_hil_control_config_rejects_unknown_backend():
    args = argparse.Namespace(
        hil_pedal_backend=None,
        hil_pedal_device=None,
        hil_pedal_debounce_s=None,
        hil_pause_code=None,
        hil_correction_code=None,
        hil_clutch_code=None,
        hil_arm_code=None,
        hil_reset_code=None,
        hil_protocol=None,
        hil_max_interventions_per_rollout=None,
    )
    with pytest.raises(ValueError, match="pedal_backend"):
        build_hil_control_config(args, {"maniflow_hil": {"pedal_backend": "mouse"}})


def test_hil_clutch_release_is_not_blocked_by_debounce():
    controls = HILControls(HILControlConfig(backend="keyboard", clutch="space", debounce_s=10.0))

    controls._set_clutch(True)
    assert controls.snapshot().clutch_active is True

    controls._set_clutch(False)
    assert controls.snapshot().clutch_active is False


def test_default_protocol_is_continuous():
    args = argparse.Namespace(hil_protocol=None, hil_max_interventions_per_rollout=None)

    cfg = build_hil_protocol_config(args, {})

    assert cfg.protocol == "continuous"
    assert cfg.max_interventions_per_rollout == 2


def test_protocol_config_reads_yaml_aliases():
    args = argparse.Namespace(hil_protocol=None, hil_max_interventions_per_rollout=None)

    cfg = build_hil_protocol_config(
        args,
        {"maniflow_hil": {"hil_protocol": "bounded", "hil_max_interventions_per_rollout": 3}},
    )

    assert cfg.protocol == "bounded"
    assert cfg.max_interventions_per_rollout == 3


def test_protocol_config_rejects_invalid_max_interventions():
    args = argparse.Namespace(hil_protocol="bounded", hil_max_interventions_per_rollout=0)

    with pytest.raises(ValueError, match="hil_max_interventions_per_rollout"):
        build_hil_protocol_config(args, {})


def test_continuous_mode_supports_multiple_corrections():
    cfg = HILProtocolConfig(protocol="continuous", max_interventions_per_rollout=2)

    assert next_phase_after_saved_correction(cfg, 1) == HILPhase.PAUSED
    assert next_phase_after_saved_correction(cfg, 2) == HILPhase.PAUSED
    assert next_phase_after_saved_correction(cfg, 3) == HILPhase.PAUSED


def test_rac_mode_terminates_after_first_saved_correction():
    cfg = HILProtocolConfig(protocol="rac", max_interventions_per_rollout=2)

    assert next_phase_after_saved_correction(cfg, 1) == HILPhase.ROLLOUT_TERMINATED


def test_bounded_mode_terminates_at_configured_intervention_count():
    cfg = HILProtocolConfig(protocol="bounded", max_interventions_per_rollout=2)

    assert next_phase_after_saved_correction(cfg, 1) == HILPhase.PAUSED
    assert next_phase_after_saved_correction(cfg, 2) == HILPhase.ROLLOUT_TERMINATED


def test_next_hil_phase_uses_post_save_intervention_count():
    phase = next_hil_phase(
        HILPhase.CORRECTING,
        snapshot(correction=True),
        protocol="bounded",
        intervention_count=1,
        max_interventions_per_rollout=2,
    )

    assert phase == HILPhase.ROLLOUT_TERMINATED


def test_bounded_limit_one_behaves_like_rac():
    cfg = HILProtocolConfig(protocol="bounded", max_interventions_per_rollout=1)

    assert next_phase_after_saved_correction(cfg, 1) == HILPhase.ROLLOUT_TERMINATED


def test_terminated_rollout_requires_reset_acknowledgement_transition():
    assert next_hil_phase(HILPhase.ROLLOUT_TERMINATED, snapshot(pause=True)) == HILPhase.ROLLOUT_TERMINATED
    assert next_hil_phase(HILPhase.ROLLOUT_TERMINATED, snapshot(correction=True)) == HILPhase.ROLLOUT_TERMINATED


def test_intervention_metadata_marks_first_and_later_interventions():
    cfg = HILProtocolConfig(protocol="bounded", max_interventions_per_rollout=2)
    rollout = HILRolloutState(rollout_id="rollout-test", intervention_count=0)

    first = correction_episode_metadata(
        protocol_config=cfg,
        rollout=rollout,
        policy_server="http://127.0.0.1:8765",
        policy_generation_id=12,
    )

    assert first["intervention_index"] == 0
    assert first["prior_human_intervention"] is False
    assert first["rollout_intervention_count"] == 1
    assert first["rollout_terminated_after_intervention"] is False
    assert first["contains_autonomous_actions"] is False
    assert first["failure_reason"] is None

    rollout.intervention_count = 1
    second = correction_episode_metadata(
        protocol_config=cfg,
        rollout=rollout,
        policy_server="http://127.0.0.1:8765",
        policy_generation_id=13,
    )

    assert second["intervention_index"] == 1
    assert second["prior_human_intervention"] is True
    assert second["rollout_intervention_count"] == 2
    assert second["rollout_terminated_after_intervention"] is True
    assert second["max_interventions_per_rollout"] == 2


def test_empty_correction_is_not_saved_or_counted(tmp_path):
    recorder = EpisodeRecorder(RecorderConfig(output_dir=tmp_path))
    recorder.start(extra={"hil_protocol": "continuous"})

    saved, samples = _finish_correction(recorder, "empty")

    assert saved is False
    assert samples == 0
    assert not list(tmp_path.glob("episode-*"))


def test_generation_gate_discards_old_responses():
    gate = HILGenerationGate()
    generation = gate.current_generation()

    gate.bump(phase=HILPhase.PAUSED)

    assert gate.accepts_response(generation) is False
    assert gate.accepts_response(gate.current_generation()) is False


def test_generation_gate_accepts_only_current_autonomous_responses():
    gate = HILGenerationGate()
    generation = gate.current_generation()

    assert gate.accepts_response(generation) is True

    next_generation = gate.bump(phase=HILPhase.AUTONOMOUS)
    assert gate.accepts_response(generation) is False
    assert gate.accepts_response(next_generation) is True


class _DelayedRuntime:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, _observation):
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return np.zeros((2, 12), dtype=np.float32)


class _Adapter:
    image_feature_keys = []
    joint_names = DEFAULT_JOINT_NAMES

    def build(self, _snapshot):
        return {}


def test_policy_worker_discards_delayed_response_from_old_generation():
    runtime = _DelayedRuntime()
    observation_buffer = LatestObservationBuffer()
    action_buffer = ActionChunkBuffer()
    armed = threading.Event()
    reset_requested = threading.Event()
    stop_requested = threading.Event()
    gate = HILGenerationGate()
    worker = PolicyWorker(
        runtime=cast(Any, runtime),
        observation_adapter=cast(Any, _Adapter()),
        observation_buffer=observation_buffer,
        action_buffer=action_buffer,
        armed=armed,
        reset_requested=reset_requested,
        stop_requested=stop_requested,
        wait_timeout_s=0.01,
        generation_provider=gate.current_generation,
        response_accepted=gate.accepts_response,
    )

    armed.set()
    worker.start()
    joints = {name: 0.0 for name in DEFAULT_JOINT_NAMES}
    observation_buffer.publish(
        timestamp_s=time.monotonic(),
        follower_state=BimanualFollowerState(
            left=joints,
            right=joints,
            read_started_monotonic_s=0.0,
            read_finished_monotonic_s=0.0,
        ),
        camera_matches={},
    )
    assert runtime.started.wait(timeout=2.0)

    gate.bump(phase=HILPhase.PAUSED)
    runtime.release.set()
    time.sleep(0.05)
    worker.stop()

    assert action_buffer.queue_length() == 0
