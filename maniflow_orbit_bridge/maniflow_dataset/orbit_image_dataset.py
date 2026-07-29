from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from termcolor import cprint

from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _to_torch_preserve_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch_preserve_strings(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


class OrbitImageDataset(BaseDataset):
    """Multi-camera Orbit zarr dataset for ManiFlow language-conditioned image policies."""

    def __init__(
        self,
        zarr_path,
        cameras=None,
        horizon=16,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=None,
        load_to_memory=False,
        **kwargs,
    ):
        super().__init__()
        self.zarr_path = str(Path(zarr_path).expanduser())
        self.cameras = list(cameras or ["overhead", "left_wrist", "right_wrist"])
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.load_to_memory = load_to_memory
        self.task_names = self._load_task_names()

        cprint(f"Loading OrbitImageDataset from {self.zarr_path}", "green")
        buffer_keys = [*self.cameras, "state", "action", "task_index"]
        if load_to_memory:
            self.replay_buffer = ReplayBuffer.copy_from_path(self.zarr_path, keys=buffer_keys)
        else:
            self.replay_buffer = ReplayBuffer.create_from_path(self.zarr_path, mode="r")
            missing = [key for key in buffer_keys if key not in self.replay_buffer]
            if missing:
                raise KeyError(f"Missing required zarr data keys: {missing}")

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        if max_train_episodes is None:
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=buffer_keys,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.train_episodes_num = int(np.sum(train_mask))
        self.val_episodes_num = int(np.sum(val_mask))

        cprint(f"Orbit cameras: {self.cameras}", "yellow")
        cprint(f"Training episodes: {self.train_episodes_num}", "yellow")
        cprint(f"Validation episodes: {self.val_episodes_num}", "yellow")

    def _load_task_names(self) -> dict[int, str]:
        sidecar_path = Path(self.zarr_path) / "orbit_tasks.json"
        if not sidecar_path.exists():
            cprint(f"No Orbit task sidecar found at {sidecar_path}; language goals will be empty", "red")
            return {}
        with sidecar_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return {int(index): str(task) for index, task in payload.get("task_names", {}).items()}

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=[*self.cameras, "state", "action", "task_index"],
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
        for camera in self.cameras:
            normalizer[camera] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = {
            "agent_pos": sample["state"].astype(np.float32),
        }
        for camera in self.cameras:
            obs[camera] = sample[camera].astype(np.float32) / 255.0

        task_index = int(np.asarray(sample["task_index"])[0])
        obs["task_name"] = self.task_names.get(task_index, "")
        return {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return _to_torch_preserve_strings(data)
