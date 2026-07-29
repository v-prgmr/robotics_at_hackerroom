"""Orbit image-only ManiFlow trainer.

ManiFlow's stock robotwin trainer imports the pointcloud policy at module import
time for type annotations. That pulls in PyTorch3D even for image-only configs.
This wrapper stubs that unused import, then reuses the stock training workspace.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types


if __name__ == "__main__":
    root_dir = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(root_dir)
    os.chdir(root_dir)


class _ImageOnlyPolicyType:
    pass


pointcloud_policy_stub = types.ModuleType("maniflow.policy.maniflow_pointcloud_policy")
setattr(pointcloud_policy_stub, "ManiFlowTransformerPointcloudPolicy", _ImageOnlyPolicyType)
sys.modules.setdefault("maniflow.policy.maniflow_pointcloud_policy", pointcloud_policy_stub)

decord_stub = types.ModuleType("decord")


class _UnavailableVideoReader:
    def __init__(self, *args, **kwargs):
        raise ImportError("decord is not installed; it is not needed for Orbit zarr training")


setattr(decord_stub, "VideoReader", _UnavailableVideoReader)
setattr(decord_stub, "cpu", lambda *args, **kwargs: None)
setattr(decord_stub, "bridge", type("_Bridge", (), {"set_bridge": staticmethod(lambda *args, **kwargs: None)})())
sys.modules.setdefault("decord", decord_stub)

import hydra

from maniflow.workspace.train_maniflow_robotwin_workspace import TrainManiFlowRoboTwinWorkspace


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
