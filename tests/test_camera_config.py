from pathlib import Path

import pytest

from bimanual_collection.hardware.cameras import CameraConfig, CameraConfigError, MultiCameraManager, resolve_camera_device


def test_resolve_camera_device_follows_symlinks(tmp_path):
    target = tmp_path / "video-target"
    target.touch()
    alias = tmp_path / "video-alias"
    alias.symlink_to(target)

    assert resolve_camera_device(str(alias)) == str(target.resolve())


def test_duplicate_camera_devices_are_rejected(tmp_path):
    target = tmp_path / "video-target"
    target.touch()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.symlink_to(target)
    right.symlink_to(target)

    with pytest.raises(CameraConfigError, match="Duplicate physical camera devices"):
        MultiCameraManager(
            [
                CameraConfig(name="left_wrist", device=str(left)),
                CameraConfig(name="right_wrist", device=str(right)),
            ]
        )
