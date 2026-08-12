from __future__ import annotations

import copy
import os
import pathlib
import sys
from collections import defaultdict

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.workspace.base_workspace import BaseWorkspace


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value.to(device, non_blocking=True) if hasattr(value, "to") else value


def _episode_ids(batch, batch_size):
    episode_id = batch.get("episode_index")
    if episode_id is None:
        raise KeyError("Progress validation requires batch['episode_index']")
    if isinstance(episode_id, torch.Tensor):
        episode_id = episode_id.detach().cpu().tolist()
    if len(episode_id) != batch_size:
        raise ValueError("episode_index must contain one identifier per batch item")
    return [str(value) for value in episode_id]


@torch.no_grad()
def evaluate_progress(policy, dataloader, device, max_steps=None):
    policy.eval()
    squared_errors = []
    absolute_errors = []
    predictions = []
    targets = []
    episode_squared_errors = defaultdict(list)

    for batch_index, batch in enumerate(dataloader):
        batch = _to_device(batch, device)
        prediction = policy.predict_progress(batch["obs"]).reshape(-1)
        target = batch["progress_target"].to(device=prediction.device, dtype=prediction.dtype).reshape(-1)
        valid = batch["progress_valid"].to(device=prediction.device, dtype=torch.bool).reshape(-1)
        ids = _episode_ids(batch, prediction.shape[0])
        error = prediction - target
        for index in valid.nonzero(as_tuple=False).reshape(-1).tolist():
            squared = float(error[index].square().item())
            squared_errors.append(squared)
            absolute_errors.append(float(error[index].abs().item()))
            predictions.append(float(prediction[index].item()))
            targets.append(float(target[index].item()))
            episode_squared_errors[ids[index]].append(squared)
        if max_steps is not None and batch_index + 1 >= max_steps:
            break

    if not squared_errors:
        return None, {}
    metrics = {
        "val_progress_mse": float(np.mean(squared_errors)),
        "val_progress_mae": float(np.mean(absolute_errors)),
        "val_progress_valid_count": len(squared_errors),
        "val_progress_pred_mean": float(np.mean(predictions)),
        "val_progress_pred_std": float(np.std(predictions)),
        "val_progress_pred_min": float(np.min(predictions)),
        "val_progress_pred_max": float(np.max(predictions)),
        "val_progress_target_mean": float(np.mean(targets)),
        "val_progress_target_std": float(np.std(targets)),
        "val_progress_target_min": float(np.min(targets)),
        "val_progress_target_max": float(np.max(targets)),
    }
    per_episode = {
        episode_id: float(np.mean(values)) for episode_id, values in episode_squared_errors.items()
    }
    return metrics, per_episode


class TrainManiFlowProgressWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        if not cfg.source_checkpoint:
            raise ValueError("source_checkpoint is required for progress-head training")
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.model.load_source_ema_checkpoint(cfg.source_checkpoint, cfg.source_state_key)
        trainable = dict(self.model.named_parameters())
        unexpected = [name for name, parameter in trainable.items() if parameter.requires_grad and not name.startswith("progress_head.")]
        if unexpected:
            raise RuntimeError(f"Only progress_head may be trainable, got: {unexpected}")
        head_parameters = [parameter for name, parameter in trainable.items() if name.startswith("progress_head.")]
        if not head_parameters or not all(parameter.requires_grad for parameter in head_parameters):
            raise RuntimeError("Every progress_head parameter must be trainable")
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=head_parameters)

        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model else None
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        dataset = hydra.utils.instantiate(cfg.progress_dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataloader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        topk = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )
        run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )

        for _ in range(cfg.training.num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_valid_count = 0
            for batch_index, batch in enumerate(train_dataloader):
                batch = _to_device(batch, device)
                loss, loss_dict = self.model.compute_loss(batch)
                valid_count = int(loss_dict["progress_valid_count"])
                self.optimizer.zero_grad(set_to_none=True)
                if valid_count:
                    loss.backward()
                    self.optimizer.step()
                    if self.ema is not None:
                        self.ema.step(self.model)
                train_loss_sum += float(loss.item()) * valid_count
                train_valid_count += valid_count
                self.global_step += 1
                if cfg.training.max_train_steps is not None and batch_index + 1 >= cfg.training.max_train_steps:
                    break

            self.epoch += 1
            log = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "train_progress_mse": train_loss_sum / train_valid_count if train_valid_count else 0.0,
                "train_progress_valid_count": train_valid_count,
            }
            if self.epoch % cfg.training.val_every == 0:
                policy = self.ema_model if self.ema_model is not None else self.model
                metrics, per_episode = evaluate_progress(
                    policy, val_dataloader, device, cfg.training.max_val_steps
                )
                if metrics is None:
                    print("Validation contained zero valid progress targets; skipping best checkpoint")
                else:
                    log.update(metrics)
                    log["val_progress_episode_mse"] = wandb.Table(
                        columns=["episode_id", "mse"], data=sorted(per_episode.items())
                    )
                    path = topk.get_ckpt_path(metrics | {"epoch": self.epoch})
                    if path is not None:
                        self.save_checkpoint(path=path, use_thread=False)
            if cfg.checkpoint.save_last_ckpt and self.epoch % cfg.training.checkpoint_every == 0:
                self.save_checkpoint(use_thread=False)
            run.log(log, step=self.global_step)


_CONFIG_ROOT = pathlib.Path(__file__).parent.parent
_CONFIG_PATH = _CONFIG_ROOT / "config"
if not _CONFIG_PATH.is_dir():
    _CONFIG_PATH = _CONFIG_ROOT / "maniflow_config"


@hydra.main(
    version_base=None,
    config_path=str(_CONFIG_PATH),
    config_name="maniflow_progress_orbit",
)
def main(cfg):
    TrainManiFlowProgressWorkspace(cfg).run()


if __name__ == "__main__":
    root_dir = str(pathlib.Path(__file__).parents[2])
    sys.path.append(root_dir)
    os.chdir(root_dir)
    main()
