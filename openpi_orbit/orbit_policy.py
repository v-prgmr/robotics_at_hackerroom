"""OpenPI policy transforms for Orbit bimanual SO-100 datasets.

Copy this file to `src/openpi/policies/orbit_policy.py` in an OpenPI checkout.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


ORBIT_ACTION_DIM = 12
ORBIT_JOINT_NAMES = (
    "left.shoulder_pan.pos",
    "left.shoulder_lift.pos",
    "left.elbow_flex.pos",
    "left.wrist_flex.pos",
    "left.wrist_roll.pos",
    "left.gripper.pos",
    "right.shoulder_pan.pos",
    "right.shoulder_lift.pos",
    "right.elbow_flex.pos",
    "right.wrist_flex.pos",
    "right.wrist_roll.pos",
    "right.gripper.pos",
)


def make_orbit_example() -> dict:
    """Create a random Orbit-format example for policy-server smoke tests."""

    return {
        "observation/state": np.random.rand(ORBIT_ACTION_DIM).astype(np.float32),
        "observation/overhead_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick one teabag and place it into the kitting tray",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Return an image as uint8 HWC RGB.

    LeRobot can hand transforms either HWC uint8 video frames or CHW float tensors depending on how the
    dataset was decoded. OpenPI model transforms expect HWC uint8 images before resizing.
    """

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {array.shape}")

    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))

    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


@dataclasses.dataclass(frozen=True)
class OrbitInputs(transforms.DataTransformFn):
    """Convert Orbit dataset or runtime observations to OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        overhead = _parse_image(data["observation/overhead_image"])
        left_wrist = _parse_image(data["observation/left_wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": overhead,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OrbitOutputs(transforms.DataTransformFn):
    """Trim OpenPI actions back to Orbit's 12-dimension bimanual action vector."""

    action_dim: int = ORBIT_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim], dtype=np.float32)}
