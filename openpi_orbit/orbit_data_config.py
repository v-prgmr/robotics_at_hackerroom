"""OpenPI training config extension for Orbit bimanual SO-100 data.

Copy this file to `src/openpi/training/orbit_config.py` in an OpenPI checkout and register
`get_orbit_configs()` from `openpi.training.config`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os

from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
from openpi.policies import orbit_policy
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import tyro
from typing_extensions import override


DEFAULT_ORBIT_REPO_ID = os.environ.get("ORBIT_LEROBOT_REPO_ID", "local/orbit_so100")
ORBIT_DELTA_ACTION_MASK = _transforms.make_bool_mask(5, -1, 5, -1)


@dataclasses.dataclass(frozen=True)
class LeRobotOrbitDataConfig(_config.DataConfigFactory):
    """Data transforms for Orbit's LeRobot v3 export.

    Orbit records absolute commanded joint targets. pi0/pi0.5 fine-tuning typically performs better with
    delta joint actions, so the five arm joints per side are converted to deltas and both grippers stay absolute.
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    action_dim: int = orbit_policy.ORBIT_ACTION_DIM
    action_sequence_keys: Sequence[str] = ("action",)
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/overhead_image": "observation.images.overhead",
                        "observation/left_wrist_image": "observation.images.left_wrist",
                        "observation/right_wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    @override
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        data_transforms = _transforms.Group(
            inputs=[orbit_policy.OrbitInputs(model_type=model_config.model_type)],
            outputs=[orbit_policy.OrbitOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_joint_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(ORBIT_DELTA_ACTION_MASK)],
                outputs=[_transforms.AbsoluteActions(ORBIT_DELTA_ACTION_MASK)],
            )

        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def get_orbit_configs() -> list[_config.TrainConfig]:
    """Return Orbit train configs to splice into OpenPI's `_CONFIGS` registry."""

    return [
        _config.TrainConfig(
            name="pi05_orbit_so100",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                discrete_state_input=False,
            ),
            data=LeRobotOrbitDataConfig(
                repo_id=DEFAULT_ORBIT_REPO_ID,
                base_config=_config.DataConfig(prompt_from_task=True),
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=100_000,
                decay_lr=5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=32,
            log_interval=100,
            save_interval=2_000,
            keep_period=10_000,
        )
    ]
