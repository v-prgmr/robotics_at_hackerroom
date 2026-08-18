import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lewm_orbit_bridge.common import episode_indices_for_split, stratified_episode_split
from lewm_orbit_bridge.convert_orbit_to_lewm import discover_episodes, load_episode_arrays
from lewm_orbit_bridge.hf_upload import HuggingFaceRunUploader, HuggingFaceUploadConfig


def test_stratified_split_is_reproducible_disjoint_and_complete():
    episodes = [
        {"episode_uid": f"{kind}/{index}", "collection_type": kind}
        for kind in ("expert", "policy_success", "policy_failure", "hil/rac_correction")
        for index in range(10)
    ]

    first = stratified_episode_split(episodes, seed=42)
    second = stratified_episode_split(episodes, seed=42)

    assert first == second
    assert {key: len(value) for key, value in first.items()} == {"train": 32, "val": 4, "test": 4}
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert set().union(*map(set, first.values())) == {episode["episode_uid"] for episode in episodes}


def test_split_consumer_rejects_episode_leakage():
    class Dataset:
        def get_episode_data(self):
            return {"episode_uid": ["a", "b", "c"]}

    manifest = {"splits": {"train": ["a", "b"], "val": ["b"], "test": ["c"]}}

    with pytest.raises(ValueError, match="overlap"):
        episode_indices_for_split(Dataset(), manifest, "train")


def _write_episode(root: Path, episode_id: str, metadata: dict) -> Path:
    episode = root / episode_id
    (episode / "videos").mkdir(parents=True)
    (episode / "videos/overhead.mp4").touch()
    (episode / "timesteps.parquet").touch()
    (episode / "episode_metadata.json").write_text(json.dumps({"episode_id": episode_id, **metadata}))
    return episode


def test_hil_saved_success_is_not_terminal_success(tmp_path):
    root = tmp_path / "maniflow_hil_bounded"
    _write_episode(root, "episode-000001", {"success": True, "collection_type": "hil"})

    episode = discover_episodes([("hil/rac_correction", root)])[0]

    assert episode["terminal_success_known"] is False
    assert episode["terminal_success"] is False
    assert episode["goal_eligible"] is False


def test_episode_conversion_preserves_duplicate_frames_and_left_right_action_order(tmp_path, monkeypatch):
    root = tmp_path / "expert"
    episode_path = _write_episode(root, "episode-000001", {})
    rows = pd.DataFrame(
        {
            "timestep_index": [4, 5, 6],
            "monotonic_timestamp_s": [1.0, 1.02, 1.04],
            "left_commanded_action": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "right_commanded_action": [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            "left_action_names": [["left_joint", "left_gripper"]] * 3,
            "right_action_names": [["right_joint", "right_gripper"]] * 3,
            "metadata_json": [json.dumps({"command_duration_s": 0.001})] * 3,
            "overhead_frame_age_s": [0.01, 0.01, 0.01],
            "overhead_video_frame_index": [9, 9, 10],
            "overhead_camera_frame_index": [100, 100, 101],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: rows)

    class Reader:
        def __init__(self, path):
            pass

        def get(self, index):
            return np.full((2, 3, 3), index, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("lewm_orbit_bridge.convert_orbit_to_lewm.SequentialVideoReader", Reader)
    episode = discover_episodes([("expert", root)])[0]

    converted, stats = load_episode_arrays(episode)

    np.testing.assert_array_equal(converted["action"][0], [1.0, 2.0, 7.0, 8.0])
    assert converted["pixels"][0][0, 0, 0] == 9
    assert converted["pixels"][1][0, 0, 0] == 9
    assert len(converted["pixels"]) == len(converted["action"]) == 3
    assert stats["action_alignment"] == "same_row_action"
    assert stats["duplicate_video_frame_fraction"] == pytest.approx(1 / 3)
    assert stats["frame_age_exceeds_command_duration_count"] == 3


def test_hf_upload_config_is_private_and_periodic_without_exposing_token():
    config = HuggingFaceUploadConfig.from_environment(
        "teabag_overhead_fs3_v1",
        {
            "HF_UPLOAD_ENABLED": "true",
            "HF_REPO_ID": "example/orbit-lewm",
            "HF_TOKEN": "secret-token",
            "HF_CHECKPOINT_INTERVAL_EPOCHS": "7",
        },
    )

    assert config.private is True
    assert config.should_upload_checkpoint(7)
    assert config.should_upload_checkpoint(14)
    assert not config.should_upload_checkpoint(6)
    assert "secret-token" not in repr(config)
    assert "token" not in config.public_metadata()


def test_hf_upload_interval_zero_disables_periodic_but_keeps_final_upload_enabled():
    config = HuggingFaceUploadConfig.from_environment(
        "run",
        {
            "HF_UPLOAD_ENABLED": "true",
            "HF_REPO_ID": "example/orbit-lewm",
            "HUGGING_FACE_HUB_TOKEN": "secret-token",
            "HF_CHECKPOINT_INTERVAL_EPOCHS": "0",
        },
    )

    assert config.enabled
    assert not config.should_upload_checkpoint(1)
    assert not config.should_upload_checkpoint(100)


def test_hf_upload_enabled_requires_repo_and_token():
    with pytest.raises(ValueError, match="HF_REPO_ID"):
        HuggingFaceUploadConfig.from_environment("run", {"HF_UPLOAD_ENABLED": "true"})

    with pytest.raises(ValueError, match="HF_TOKEN"):
        HuggingFaceUploadConfig.from_environment(
            "run", {"HF_UPLOAD_ENABLED": "true", "HF_REPO_ID": "example/orbit-lewm"}
        )


def test_hf_private_upload_refuses_existing_public_repo(monkeypatch):
    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, **kwargs):
            return types.SimpleNamespace(private=False)

    fake_module = types.SimpleNamespace(HfApi=FakeApi, create_repo=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    config = HuggingFaceUploadConfig.from_environment(
        "run",
        {
            "HF_UPLOAD_ENABLED": "true",
            "HF_REPO_ID": "example/public-repo",
            "HF_TOKEN": "secret-token",
            "HF_UPLOAD_RETRY_DELAY_S": "0",
        },
    )

    with pytest.raises(ValueError, match="is public"):
        HuggingFaceRunUploader(config)


def test_train_launcher_uses_package_module_from_any_working_directory(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$PYTHONPATH\" \"$@\"\n")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "ORBIT_DIR": str(repo_root),
        "LEWM_PYTHON": str(fake_python),
        "PYTHONPATH": "/existing/pythonpath",
    }

    result = subprocess.run(
        [str(repo_root / "lewm_orbit_bridge/train_teabag_lewm.sh"), "--mode", "smoke"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = result.stdout.splitlines()
    assert lines[0] == f"{repo_root}:/existing/pythonpath"
    assert lines[1:3] == ["-m", "lewm_orbit_bridge.train_lewm"]


def test_training_config_requires_explicit_dataset_paths_and_uses_spt_manager():
    repo_root = Path(__file__).resolve().parents[1]
    config = (repo_root / "lewm_orbit_bridge/config/teabag.yaml").read_text()
    trainer = (repo_root / "lewm_orbit_bridge/train_lewm.py").read_text()

    assert "dataset_path: ${oc.env:LEWM_DATASET}" in config
    assert "split_manifest: ${oc.env:LEWM_SPLITS}" in config
    assert "~/.stable-wm" not in config
    assert "manager = spt.Manager(" in trainer
    assert "manager()" in trainer
