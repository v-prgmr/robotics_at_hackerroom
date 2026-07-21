from bimanual_collection.hardware.bimanual_robot import BimanualRobot, BimanualRobotConfig


class FakeFollower:
    def __init__(self, start):
        self.state = dict(start)
        self.is_connected = True
        self.commands = []

    def connect(self, calibrate: bool = True) -> None:
        self.is_connected = True

    def get_observation(self):
        return dict(self.state)

    def send_action(self, action):
        self.state.update(action)
        self.commands.append(dict(action))
        return dict(action)

    def disconnect(self) -> None:
        self.is_connected = False


def test_move_to_positions_interpolates_to_targets():
    left = FakeFollower({"shoulder_pan.pos": 0.0, "gripper.pos": 0.0})
    right = FakeFollower({"shoulder_pan.pos": 10.0, "gripper.pos": 10.0})
    robot = BimanualRobot(BimanualRobotConfig("left", "right"), left_arm=left, right_arm=right)

    robot.move_to_positions(
        {"shoulder_pan.pos": 2.0, "gripper.pos": 4.0},
        {"shoulder_pan.pos": 12.0, "gripper.pos": 14.0},
        duration_s=0.0,
        steps=2,
    )

    assert left.commands == [
        {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0},
        {"shoulder_pan.pos": 2.0, "gripper.pos": 4.0},
    ]
    assert right.commands == [
        {"shoulder_pan.pos": 11.0, "gripper.pos": 12.0},
        {"shoulder_pan.pos": 12.0, "gripper.pos": 14.0},
    ]
