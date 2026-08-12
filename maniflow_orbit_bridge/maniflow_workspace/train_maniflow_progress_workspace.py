from __future__ import annotations

import copy
import os
import pathlib
import sys
from collections import defaultdict

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


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


def _divide_gradients(parameters, divisor):
    if divisor <= 0:
        raise ValueError("Gradient divisor must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


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
    include_keys = ("global_step", "optimizer_step", "epoch")

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
        self.head_parameters = [
            parameter for name, parameter in trainable.items() if name.startswith("progress_head.")
        ]
        if not self.head_parameters or not all(parameter.requires_grad for parameter in self.head_parameters):
            raise RuntimeError("Every progress_head parameter must be trainable")
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.head_parameters)

        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model else None
        self.global_step = 0
        self.optimizer_step = 0
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

        num_grad_steps = cfg.training.num_grad_steps
        if num_grad_steps is not None:
            num_grad_steps = int(num_grad_steps)
            if num_grad_steps <= 0:
                raise ValueError("training.num_grad_steps must be positive when configured")
        gradient_accumulate_every = int(cfg.training.gradient_accumulate_every)
        if gradient_accumulate_every <= 0:
            raise ValueError("training.gradient_accumulate_every must be positive")

        def training_complete():
            if num_grad_steps is not None:
                return self.optimizer_step >= num_grad_steps
            return self.epoch >= int(cfg.training.num_epochs)

        accumulation_microbatches = 0
        accumulation_valid_count = 0
        accumulation_loss_sum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        progress_total = num_grad_steps
        progress = tqdm.tqdm(
            total=progress_total,
            initial=self.optimizer_step if progress_total is not None else 0,
            desc="Progress optimizer steps",
            unit="step",
            dynamic_ncols=True,
        )
        try:
            while not training_complete():
                self.model.train()
                train_loss_sum = 0.0
                train_valid_count = 0
                for batch_index, batch in enumerate(train_dataloader):
                    batch = _to_device(batch, device)
                    loss, loss_dict = self.model.compute_loss(batch)
                    valid_count = int(loss_dict["progress_valid_count"])
                    optimizer_stepped = False
                    step_loss_sum = 0.0
                    if valid_count:
                        step_loss_sum = float(loss.item()) * valid_count
                        (loss * valid_count).backward()
                        accumulation_microbatches += 1
                        accumulation_valid_count += valid_count
                        accumulation_loss_sum += step_loss_sum
                        if accumulation_microbatches == gradient_accumulate_every:
                            accumulated_mse = accumulation_loss_sum / accumulation_valid_count
                            _divide_gradients(self.head_parameters, accumulation_valid_count)
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self.optimizer_step += 1
                            optimizer_stepped = True
                            accumulation_microbatches = 0
                            accumulation_valid_count = 0
                            accumulation_loss_sum = 0.0
                            if self.ema is not None:
                                self.ema.step(self.model)
                            progress.update(1)
                            progress.set_postfix(
                                loss=f"{accumulated_mse:.6f}",
                                epoch=self.epoch,
                                microbatch=self.global_step + 1,
                            )
                            run.log(
                                {
                                    "train_progress_mse_step": accumulated_mse,
                                    "train_progress_valid_count_step": train_valid_count + valid_count,
                                    "optimizer_step": self.optimizer_step,
                                    "global_step": self.global_step + 1,
                                    "epoch": self.epoch,
                                },
                                step=self.optimizer_step,
                            )
                    train_loss_sum += step_loss_sum
                    train_valid_count += valid_count
                    self.global_step += 1
                    if training_complete():
                        break
                    if cfg.training.max_train_steps is not None and batch_index + 1 >= cfg.training.max_train_steps:
                        break

                if train_valid_count == 0:
                    raise RuntimeError("Training epoch contained zero valid progress targets")
                self.epoch += 1
                final_epoch = num_grad_steps is None and self.epoch >= int(cfg.training.num_epochs)
                if final_epoch and accumulation_microbatches:
                    _divide_gradients(self.head_parameters, accumulation_valid_count)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer_step += 1
                    accumulation_microbatches = 0
                    accumulation_valid_count = 0
                    accumulation_loss_sum = 0.0
                    if self.ema is not None:
                        self.ema.step(self.model)
                    progress.update(1)
                log = {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "gradient_accumulate_every": gradient_accumulate_every,
                    "pending_accumulation_microbatches": accumulation_microbatches,
                    "train_progress_mse": train_loss_sum / train_valid_count,
                    "train_progress_valid_count": train_valid_count,
                }
                final_step = training_complete()
                accumulation_complete = accumulation_microbatches == 0
                if accumulation_complete and (self.epoch % cfg.training.val_every == 0 or final_step):
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
                if accumulation_complete and cfg.checkpoint.save_last_ckpt and (
                    self.epoch % cfg.training.checkpoint_every == 0 or final_step
                ):
                    self.save_checkpoint(use_thread=False)
                run.log(log, step=self.optimizer_step)
        finally:
            progress.close()


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
