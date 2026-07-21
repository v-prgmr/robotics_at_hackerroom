import time

from bimanual_collection.control.clutch import BimanualClutchController
from bimanual_collection.hardware.pedals import FootSwitchConfig, FootSwitchManager


def pose(base: float) -> dict[str, float]:
    return {
        "shoulder_pan.pos": base,
        "shoulder_lift.pos": base + 1,
        "elbow_flex.pos": base + 2,
        "wrist_flex.pos": base + 3,
        "wrist_roll.pos": base + 4,
        "gripper.pos": base + 5,
    }


def test_left_clutch_holds_left_while_right_continues():
    controller = BimanualClutchController()
    first = controller.update(pose(0), pose(10), pose(100), pose(200), False, False, False)
    assert first.left_action == pose(100)
    assert first.right_action == pose(200)

    moved = controller.update(pose(1), pose(11), pose(100), pose(200), True, False, False)
    assert moved.left_action == pose(100)
    assert moved.right_action == pose(201)
    assert not moved.should_record
    assert [event.event for event in moved.events] == ["clutch_engaged"]


def test_leader_absolute_startup_targets_leader_pose():
    controller = BimanualClutchController(startup_alignment="leader_absolute")

    first = controller.update(pose(0), pose(10), pose(100), pose(200), False, False, False)

    assert first.left_action == pose(0)
    assert first.right_action == pose(10)


def test_hold_current_startup_targets_current_follower_pose():
    controller = BimanualClutchController(startup_alignment="hold_current")

    first = controller.update(pose(0), pose(10), pose(100), pose(200), False, False, False)

    assert first.left_action == pose(100)
    assert first.right_action == pose(200)


def test_releasing_one_clutch_resets_only_that_side():
    controller = BimanualClutchController()
    controller.update(pose(0), pose(10), pose(100), pose(200), False, False, False)
    controller.update(pose(5), pose(11), pose(100), pose(201), True, False, False)

    released = controller.update(pose(9), pose(12), pose(103), pose(202), False, False, False)
    assert released.left_action == pose(103)
    assert released.right_action == pose(202)
    assert [event.event for event in released.events] == ["clutch_released"]

    moved_again = controller.update(pose(10), pose(13), pose(103), pose(202), False, False, False)
    assert moved_again.left_action == pose(104)
    assert moved_again.right_action == pose(203)


def test_pause_holds_both_and_resume_resets_references():
    controller = BimanualClutchController()
    controller.update(pose(0), pose(10), pose(100), pose(200), False, False, False)
    moving = controller.update(pose(1), pose(11), pose(100), pose(200), False, False, False)
    assert moving.left_action == pose(101)
    assert moving.right_action == pose(201)

    paused = controller.update(pose(9), pose(19), pose(105), pose(205), False, False, True)
    assert paused.left_action == pose(101)
    assert paused.right_action == pose(201)
    assert not paused.should_record

    resumed = controller.update(pose(20), pose(30), pose(110), pose(210), False, False, False)
    assert resumed.left_action == pose(110)
    assert resumed.right_action == pose(210)
    assert resumed.should_record


def test_pause_toggle_edge_is_consumed_once():
    manager = FootSwitchManager(FootSwitchConfig(enabled=True, debounce_s=0.0))
    manager._toggle_pause()

    first = manager.snapshot()
    second = manager.snapshot()
    assert first.recording_paused
    assert first.pause_edge
    assert second.recording_paused
    assert not second.pause_edge


def test_recording_third_pedal_short_press_emits_start_save_edge():
    manager = FootSwitchManager(
        FootSwitchConfig(enabled=True, debounce_s=0.0, third_pedal_mode="recording", recording_hold_cancel_s=1.0)
    )

    manager._set_recording_control(True)
    manager._set_recording_control(False)

    first = manager.snapshot()
    second = manager.snapshot()
    assert first.recording_start_save_edge
    assert not first.recording_cancel_edge
    assert not second.recording_start_save_edge


def test_recording_third_pedal_hold_emits_cancel_edge_only():
    manager = FootSwitchManager(
        FootSwitchConfig(enabled=True, debounce_s=0.0, third_pedal_mode="recording", recording_hold_cancel_s=0.01)
    )

    manager._set_recording_control(True)
    time.sleep(0.05)
    held = manager.snapshot()
    manager._set_recording_control(False)
    released = manager.snapshot()

    assert not held.recording_start_save_edge
    assert held.recording_cancel_edge
    assert not released.recording_start_save_edge
    assert not released.recording_cancel_edge
