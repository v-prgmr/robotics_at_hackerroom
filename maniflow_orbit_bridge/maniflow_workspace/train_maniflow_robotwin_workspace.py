if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

import os
import json
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Sampler
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys
sys.path.insert(0, '../')
sys.path.append('maniflow/env_runner')
sys.path.append('maniflow/policy')

from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class BalancedWindowBatchSampler(Sampler):
    """Yields mixed full-demo/RaC microbatches with fixed per-source counts."""

    def __init__(
        self,
        *,
        full_len,
        rac_len,
        full_batch_size,
        rac_batch_size,
        num_batches,
        seed,
        start_batch=0,
        shuffle_combined=True,
    ):
        batch_size = int(full_batch_size) + int(rac_batch_size)
        if batch_size % 2 != 0:
            raise ValueError(f"LoRA/RaC mixing requires an even microbatch size, got {batch_size}")
        if full_batch_size <= 0 or rac_batch_size <= 0:
            raise ValueError(
                f"Both sources must contribute to every microbatch, got full={full_batch_size}, rac={rac_batch_size}"
            )
        if full_len <= 0 or rac_len <= 0:
            raise ValueError(f"Both datasets must contain training windows, got full={full_len}, rac={rac_len}")
        if num_batches < 0:
            raise ValueError(f"num_batches must be non-negative, got {num_batches}")
        if start_batch < 0 or start_batch > num_batches:
            raise ValueError(f"start_batch must be in [0, {num_batches}], got {start_batch}")
        self.full_len = int(full_len)
        self.rac_len = int(rac_len)
        self.full_batch_size = int(full_batch_size)
        self.rac_batch_size = int(rac_batch_size)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.shuffle_combined = bool(shuffle_combined)

    def __len__(self):
        return self.num_batches - self.start_batch

    def state_dict(self):
        return {
            'full_len': self.full_len,
            'rac_len': self.rac_len,
            'full_batch_size': self.full_batch_size,
            'rac_batch_size': self.rac_batch_size,
            'batch_size': self.batch_size,
            'num_batches': self.num_batches,
            'seed': self.seed,
            'start_batch': self.start_batch,
            'shuffle_combined': self.shuffle_combined,
        }

    def load_state_dict(self, state_dict):
        self.full_len = int(state_dict['full_len'])
        self.rac_len = int(state_dict['rac_len'])
        self.full_batch_size = int(state_dict.get('full_batch_size', int(state_dict['batch_size']) // 2))
        self.rac_batch_size = int(state_dict.get('rac_batch_size', int(state_dict['batch_size']) - self.full_batch_size))
        self.batch_size = int(state_dict['batch_size'])
        self.num_batches = int(state_dict['num_batches'])
        self.seed = int(state_dict['seed'])
        self.start_batch = int(state_dict['start_batch'])
        self.shuffle_combined = bool(state_dict['shuffle_combined'])

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        full_order = np.arange(self.full_len)
        rac_order = np.arange(self.rac_len)
        full_pos = self.full_len
        rac_pos = self.rac_len

        def draw(order, pos, count):
            chunks = []
            remaining = count
            while remaining > 0:
                if pos >= len(order):
                    rng.shuffle(order)
                    pos = 0
                take = min(remaining, len(order) - pos)
                chunks.append(order[pos:pos + take])
                pos += take
                remaining -= take
            return np.concatenate(chunks), pos

        for batch_idx in range(self.num_batches):
            full_idx, full_pos = draw(full_order, full_pos, self.full_batch_size)
            rac_idx, rac_pos = draw(rac_order, rac_pos, self.rac_batch_size)
            rac_idx = rac_idx + self.full_len
            batch = np.concatenate([full_idx, rac_idx])
            if self.shuffle_combined:
                rng.shuffle(batch)
            if batch_idx < self.start_batch:
                continue
            yield [int(idx) for idx in batch]


def _compute_source_batch_sizes(batch_size, full_fraction, rac_fraction):
    batch_size = int(batch_size)
    if batch_size % 2 != 0:
        raise ValueError(f"LoRA/RaC mixing requires an even microbatch size, got {batch_size}")
    full_fraction = float(full_fraction)
    rac_fraction = float(rac_fraction)
    if full_fraction <= 0 or rac_fraction <= 0:
        raise ValueError(
            f"full_fraction and rac_fraction must both be positive, got {full_fraction}, {rac_fraction}"
        )
    fraction_sum = full_fraction + rac_fraction
    full_batch_size = int(round(batch_size * full_fraction / fraction_sum))
    full_batch_size = min(max(full_batch_size, 1), batch_size - 1)
    rac_batch_size = batch_size - full_batch_size
    return full_batch_size, rac_batch_size


def _is_lora_rac_finetune(cfg):
    return OmegaConf.select(cfg, 'finetune.mode', default='dense') == 'lora_rac'


def _lora_enabled(cfg):
    return bool(OmegaConf.select(cfg, 'finetune.lora.enabled', default=False))


def _count_trainable_parameters(module):
    trainable = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for _, param in trainable)
    total_count = sum(param.numel() for param in module.parameters())
    return trainable, trainable_count, total_count


def _apply_head_lora(policy, lora_cfg):
    from peft import LoraConfig, get_peft_model

    target_modules = list(lora_cfg.target_modules)
    expected_target_count = int(lora_cfg.expected_target_count)
    named_modules = dict(policy.model.named_modules())
    targeted_names = []
    for module_name in target_modules:
        module = named_modules.get(module_name)
        if module is None:
            raise ValueError(f"LoRA target module not found under policy.model: {module_name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target must be nn.Linear, got {module_name}: {type(module)}")
        targeted_names.append(module_name)
    if len(targeted_names) != expected_target_count:
        raise ValueError(
            f"Expected {expected_target_count} LoRA targets, matched {len(targeted_names)}: {targeted_names}"
        )

    for param in policy.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=int(lora_cfg.rank),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        bias=str(lora_cfg.bias),
        target_modules=target_modules,
    )
    policy.model = get_peft_model(policy.model, peft_config)
    actual_targeted = list(getattr(policy.model, 'targeted_module_names', targeted_names))
    if sorted(actual_targeted) != sorted(targeted_names):
        raise RuntimeError(f"PEFT targeted unexpected modules: expected={targeted_names}, actual={actual_targeted}")

    trainable, trainable_count, total_count = _count_trainable_parameters(policy)
    bad_trainable = [name for name, _ in trainable if 'lora_' not in name]
    if bad_trainable:
        raise RuntimeError(f"Only LoRA parameters may be trainable, found: {bad_trainable[:20]}")
    if trainable_count == 0:
        raise RuntimeError("LoRA injection produced zero trainable parameters")

    percent = 100.0 * trainable_count / max(total_count, 1)
    cprint(f"[LoRA] Targeted policy.model modules: {targeted_names}", "cyan")
    cprint(
        f"[LoRA] Trainable params: {trainable_count:,} / {total_count:,} ({percent:.4f}%)",
        "cyan",
    )
    return targeted_names


def _apply_lora_with_optional_zero_check(policy, cfg):
    lora_cfg = cfg.finetune.lora
    if not bool(OmegaConf.select(cfg, 'finetune.lora.verify_zero_init', default=False)):
        return _apply_head_lora(policy, lora_cfg)

    dense_model = policy.model
    dense_model.eval()
    device = next(dense_model.parameters()).device
    batch_size = 1
    sample = torch.randn(
        batch_size,
        dense_model.horizon,
        dense_model.input_emb.in_features,
        device=device,
    )
    timestep = torch.full((batch_size,), 0.25, device=device)
    target_t = torch.full((batch_size,), 0.5, device=device)
    vis_len = dense_model.visual_cond_len * int(dense_model.n_obs_steps or 1)
    vis_cond = torch.randn(
        batch_size,
        vis_len,
        dense_model.vis_cond_obs_emb.in_features,
        device=device,
    )
    lang_cond = [""] * batch_size if dense_model.language_conditioned else None

    with torch.no_grad():
        dense_out = dense_model(
            sample=sample,
            timestep=timestep,
            target_t=target_t,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
        ).detach()

    targeted_names = _apply_head_lora(policy, lora_cfg)
    policy.model.eval()
    with torch.no_grad():
        lora_out = policy.model(
            sample=sample,
            timestep=timestep,
            target_t=target_t,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
        ).detach()
    torch.testing.assert_close(lora_out, dense_out, rtol=1e-5, atol=1e-6)
    cprint("[LoRA] Zero-initialized adapter output matches dense base output", "cyan")
    return targeted_names


def _evaluate_mean_loss(model, ema_model, dataloader, *, device, cfg, desc):
    loss_sum = 0.0
    window_count = 0
    with torch.no_grad():
        with tqdm.tqdm(dataloader, desc=desc, leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss, _ = model.compute_loss(batch, ema_model)
                batch_size = int(batch['action'].shape[0])
                loss_sum += float(loss.detach().cpu()) * batch_size
                window_count += batch_size
                if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                    break
    if window_count == 0:
        return None
    return loss_sum / window_count

class TrainManiFlowRoboTwinWorkspace:
    include_keys = ['global_step', 'epoch', 'optimizer_step', 'micro_step', 'epoch_micro_step', 'sampler_state', 'rng_state']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        self._pending_state_dicts = {}
        self.sampler_state = None
        self.rng_state = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)
        self.lora_targeted_modules = []

        init_from_checkpoint = OmegaConf.select(cfg, 'finetune.init_from_checkpoint')
        explicit_resume = OmegaConf.select(cfg, 'training.resume_from_checkpoint')
        if init_from_checkpoint and explicit_resume:
            raise ValueError(
                "finetune.init_from_checkpoint initializes fresh adapters and cannot be combined "
                "with training.resume_from_checkpoint. Use resume_from_checkpoint only for continuing "
                "an existing LoRA run."
            )
        if init_from_checkpoint:
            self._load_dense_init_checkpoint(init_from_checkpoint, OmegaConf.select(cfg, 'finetune.init_state_key', default='ema_model'))

        if _lora_enabled(cfg):
            self.lora_targeted_modules = _apply_lora_with_optional_zero_check(self.model, cfg)

        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        optimizer_params = [param for param in self.model.parameters() if param.requires_grad]
        if len(optimizer_params) == 0:
            raise RuntimeError("Optimizer received zero trainable parameters")
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=optimizer_params)
        # self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.optimizer_step = 0
        self.micro_step = 0
        self.epoch_micro_step = 0
        self.epoch = 0

    def _capture_rng_state(self):
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self):
        if not self.rng_state:
            return
        random.setstate(self.rng_state['python'])
        np.random.set_state(self.rng_state['numpy'])
        torch.set_rng_state(self.rng_state['torch'])
        if self.rng_state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.rng_state['cuda'])

    def _load_pending_state_dicts(self):
        for key in list(self._pending_state_dicts.keys()):
            if key in self.__dict__ and hasattr(self.__dict__[key], 'load_state_dict'):
                self.__dict__[key].load_state_dict(self._pending_state_dicts.pop(key))

    def _load_dense_init_checkpoint(self, path, state_key):
        ckpt_path = pathlib.Path(str(path)).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Dense init checkpoint does not exist: {ckpt_path}")
        payload = torch.load(ckpt_path.open('rb'), pickle_module=dill, map_location='cpu')
        state_dicts = payload.get('state_dicts', {})
        if state_key not in state_dicts:
            raise KeyError(f"Checkpoint has no '{state_key}' state dict; available keys: {sorted(state_dicts)}")
        self.model.load_state_dict(state_dicts[state_key], strict=True)
        cprint(f"Initialized dense policy weights from {ckpt_path} [{state_key}]", "cyan")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        WANDB = True

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        RUN_ROLLOUT = False
        RUN_VALIDATION = True # reduce time cost

        # resume training
        resume_from_checkpoint = OmegaConf.select(cfg, 'training.resume_from_checkpoint')
        init_from_checkpoint = OmegaConf.select(cfg, 'finetune.init_from_checkpoint')
        should_resume = (bool(cfg.training.resume) and not bool(init_from_checkpoint)) or bool(resume_from_checkpoint)
        if should_resume:
            if resume_from_checkpoint:
                lastest_ckpt_path = pathlib.Path(str(resume_from_checkpoint)).expanduser()
                if not lastest_ckpt_path.is_file():
                    raise FileNotFoundError(f"Resume checkpoint does not exist: {lastest_ckpt_path}")
            else:
                lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                print(f"Loaded checkpoint epoch={self.epoch}, global_step={self.global_step}")
                if bool(OmegaConf.select(cfg, 'training.advance_epoch_on_resume', default=False)):
                    self.epoch += 1
                    self.global_step += 1
                    print(f"Advanced resume position to epoch={self.epoch}, global_step={self.global_step}")
            else:
                print(f"No checkpoint found at {lastest_ckpt_path}; starting from scratch")

        target_epoch = OmegaConf.select(cfg, 'training.target_epoch')
        if target_epoch is not None:
            target_epoch = int(target_epoch)
            if self.epoch > target_epoch:
                print(f"Current epoch {self.epoch} is greater than target_epoch {target_epoch}; nothing to train")
                return

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.robotwin_task.dataset)
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")

        rac_dataset = None
        val_full_dataloader = None
        val_rac_dataloader = None
        train_dataset = None
        lora_microbatches_per_epoch = None
        lora_rac_finetune = _is_lora_rac_finetune(cfg)
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)

        if lora_rac_finetune:
            rac_dataset_cfg = copy.deepcopy(cfg.robotwin_task.dataset)
            rac_dataset_cfg.zarr_path = cfg.finetune.data.rac_zarr_path
            rac_dataset = hydra.utils.instantiate(rac_dataset_cfg)
            assert isinstance(rac_dataset, BaseDataset), print(f"rac_dataset must be BaseDataset, got {type(rac_dataset)}")

            microbatch_size = int(dataloader_cfg.pop('batch_size'))
            dataloader_cfg.pop('shuffle', None)
            dataloader_cfg.pop('drop_last', None)
            full_batch_size, rac_batch_size = _compute_source_batch_sizes(
                microbatch_size,
                cfg.finetune.data.full_fraction,
                cfg.finetune.data.rac_fraction,
            )
            source_microbatches = int(np.ceil(max(
                len(dataset) / full_batch_size,
                len(rac_dataset) / rac_batch_size,
            )))
            lora_microbatches_per_epoch = int(
                np.ceil(source_microbatches / int(cfg.training.gradient_accumulate_every))
                * int(cfg.training.gradient_accumulate_every)
            )
            full_windows_per_epoch = lora_microbatches_per_epoch * full_batch_size
            rac_windows_per_epoch = lora_microbatches_per_epoch * rac_batch_size
            full_repeats = full_windows_per_epoch / len(dataset)
            rac_repeats = rac_windows_per_epoch / len(rac_dataset)
            full_oversample = full_windows_per_epoch - len(dataset)
            rac_oversample = rac_windows_per_epoch - len(rac_dataset)
            train_dataset = ConcatDataset([dataset, rac_dataset])
            train_dataloader = None
            cprint(
                "LoRA/RaC epoch semantics: one epoch follows the configured full/RaC fraction "
                "and runs until both sources are approximately covered once; a source is "
                "reshuffled and recycled only if its configured allocation exhausts it.",
                'cyan',
            )
            cprint(
                f"LoRA/RaC windows: full={len(dataset)}, rac={len(rac_dataset)}, "
                f"microbatch allocation={full_batch_size} full + {rac_batch_size} rac "
                f"(requested fractions={float(cfg.finetune.data.full_fraction):.4f}/"
                f"{float(cfg.finetune.data.rac_fraction):.4f})",
                'cyan',
            )
            cprint(
                f"LoRA/RaC epoch size: {lora_microbatches_per_epoch} microbatches, "
                f"{lora_microbatches_per_epoch // int(cfg.training.gradient_accumulate_every)} optimizer steps, "
                f"full seen={full_windows_per_epoch} ({full_repeats:.3f}x, oversample={full_oversample}), "
                f"rac seen={rac_windows_per_epoch} ({rac_repeats:.3f}x, oversample={rac_oversample})",
                'cyan',
            )
            cprint(
                "If the RaC dataset is small, monitor val/rac_loss for overfitting from repeated samples.",
                'yellow',
            )

            val_dataloader_cfg['drop_last'] = False
            val_dataloader_cfg['shuffle'] = False
            val_full_dataloader = DataLoader(dataset.get_validation_dataset(), **val_dataloader_cfg)
            val_rac_dataloader = DataLoader(rac_dataset.get_validation_dataset(), **val_dataloader_cfg)
            val_dataloader = val_full_dataloader
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()

        # print dataset info
        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
        cprint(f"Number of validation episodes: {dataset.val_episodes_num}", 'red')
        if rac_dataset is not None:
            cprint(f"RaC Dataset Path: {rac_dataset.zarr_path}", 'red')
            cprint(f"RaC training episodes: {rac_dataset.train_episodes_num}", 'red')
            cprint(f"RaC validation episodes: {rac_dataset.val_episodes_num}", 'red')


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_num_epochs = OmegaConf.select(cfg, 'training.lr_num_epochs')
        if lr_num_epochs is None:
            lr_num_epochs = cfg.training.num_epochs
        lr_num_epochs = int(lr_num_epochs)
        gradient_accumulate_every = int(cfg.training.gradient_accumulate_every)
        if lora_rac_finetune:
            scheduler_step = int(self.optimizer_step)
            num_training_steps = (lora_microbatches_per_epoch * lr_num_epochs) // cfg.training.gradient_accumulate_every
        else:
            scheduler_step = (int(self.global_step) + gradient_accumulate_every - 1) // gradient_accumulate_every
            num_training_steps = (len(train_dataloader) * lr_num_epochs) // cfg.training.gradient_accumulate_every
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=scheduler_step-1
        )
        print(
            f"LR scheduler resume: lr_num_epochs={lr_num_epochs}, "
            f"num_training_steps={num_training_steps}, scheduler_step={scheduler_step}, "
            f"lr={self.lr_scheduler.get_last_lr()[0]:.8g}"
        )

        # configure ema
        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        self._load_pending_state_dicts()

        # configure env runner
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.robotwin_task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseRunner)

        env_runner = None

        cfg.logging.name = str(cfg.robotwin_task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.optimizer.zero_grad(set_to_none=True)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        if target_epoch is None:
            epochs_to_run = cfg.training.num_epochs
        else:
            epochs_to_run = target_epoch - self.epoch + 1
        print(f"Training for {epochs_to_run} epoch(s), starting at epoch {self.epoch}")
        for local_epoch_idx in range(epochs_to_run):
            step_log = dict()
            # ========= train for this epoch ==========
            self.model.train()
            if self.ema_model is not None:
                self.ema_model.eval()
            train_losses = list()
            if lora_rac_finetune:
                sampler_seed = int(cfg.training.seed) + int(self.epoch)
                sampler_start_batch = int(self.epoch_micro_step)
                if self.sampler_state is not None:
                    sampler_seed = int(self.sampler_state.get('seed', sampler_seed))
                    sampler_start_batch = int(self.sampler_state.get('start_batch', sampler_start_batch))
                train_sampler = BalancedWindowBatchSampler(
                    full_len=len(dataset),
                    rac_len=len(rac_dataset),
                    full_batch_size=full_batch_size,
                    rac_batch_size=rac_batch_size,
                    num_batches=lora_microbatches_per_epoch,
                    seed=sampler_seed,
                    start_batch=sampler_start_batch,
                    shuffle_combined=bool(cfg.finetune.data.shuffle_combined_microbatch),
                )
                self.sampler_state = train_sampler.state_dict()
                train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, **dataloader_cfg)
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()

                    # Forward pass
                    raw_loss, loss_dict = self.model.compute_loss(batch, self.ema_model)


                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    accumulation_window_complete = (
                        (self.micro_step + 1) % cfg.training.gradient_accumulate_every == 0
                    )
                    # step optimizer only after the full gradient-accumulation window
                    if accumulation_window_complete:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        if cfg.training.use_ema:
                            self.ema.step(self.model)
                        self.optimizer.zero_grad(set_to_none=True)
                        self.optimizer_step += 1
                    t1_3 = time.time()
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'micro_step': self.micro_step,
                        'optimizer_step': self.optimizer_step,
                        'epoch': self.epoch,
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    self.micro_step += 1
                    self.epoch_micro_step += 1
                    if lora_rac_finetune:
                        self.sampler_state = {
                            **train_sampler.state_dict(),
                            'start_batch': int(self.epoch_micro_step),
                        }
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            if len(train_losses) == 0:
                print("No training microbatches remaining; exiting training loop")
                return
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            epoch_complete = True
            if lora_rac_finetune:
                epoch_complete = int(self.epoch_micro_step) >= int(lora_microbatches_per_epoch)
            completed_epoch = self.epoch + 1
            checkpoint_due = (
                epoch_complete
                and cfg.checkpoint.save_ckpt
                and (completed_epoch % cfg.training.checkpoint_every) == 0
            )
            validation_due = (
                RUN_VALIDATION
                and epoch_complete
                and (
                    (completed_epoch % cfg.training.val_every) == 0
                    or checkpoint_due
                )
            )
            step_log['epoch'] = completed_epoch

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if cfg.training.debug:
                min_epoch_rollout = 0
            else:
                min_epoch_rollout = 300
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.epoch >= min_epoch_rollout: # and self.epoch > 1, and self.epoch >= 100
                cprint(f"Running rollout for epoch {self.epoch}", 'cyan')
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)
            elif self.epoch == 0:
                runner_log = dict()
                runner_log['test_mean_score'] = 0
                runner_log['mean_success_rates'] = 0
                runner_log['SR_test_L3'] = 0
                runner_log['SR_test_L5'] = 0
                runner_log['sim_video_eval'] = None
                step_log.update(runner_log)

            # run validation
            if validation_due:
                self.model.eval()
                if self.ema_model is not None:
                    self.ema_model.eval()
                if lora_rac_finetune:
                    val_full_loss = _evaluate_mean_loss(
                        self.model,
                        self.ema_model,
                        val_full_dataloader,
                        device=device,
                        cfg=cfg,
                        desc=f"Validation full epoch {completed_epoch}",
                    )
                    val_rac_loss = _evaluate_mean_loss(
                        self.model,
                        self.ema_model,
                        val_rac_dataloader,
                        device=device,
                        cfg=cfg,
                        desc=f"Validation RaC epoch {completed_epoch}",
                    )
                    if val_full_loss is not None and val_rac_loss is not None:
                        val_combined_loss = 0.5 * val_full_loss + 0.5 * val_rac_loss
                        step_log['val/full_loss'] = val_full_loss
                        step_log['val/rac_loss'] = val_rac_loss
                        step_log['val/combined_50_50_loss'] = val_combined_loss
                        # Existing Top-K manager defaults to val_loss; mirror the RaC metric for selection.
                        step_log['val_loss'] = val_rac_loss
                        cprint(
                            f"Validation losses: full={val_full_loss:.6f}, "
                            f"rac={val_rac_loss:.6f}, combined_50_50={val_combined_loss:.6f}",
                            'cyan',
                        )
                else:
                    val_loss = _evaluate_mean_loss(
                        self.model,
                        self.ema_model,
                        val_dataloader,
                        device=device,
                        cfg=cfg,
                        desc=f"Validation epoch {completed_epoch}",
                    )
                    if val_loss is not None:
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']

                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None or step_log.get('test_mean_score', None) is None:
                step_log['test_mean_score'] = - train_loss

            if epoch_complete:
                self.epoch = completed_epoch
                self.epoch_micro_step = 0
                if lora_rac_finetune:
                    self.sampler_state = {
                        'full_len': len(dataset),
                        'rac_len': len(rac_dataset),
                        'full_batch_size': full_batch_size,
                        'rac_batch_size': rac_batch_size,
                        'batch_size': microbatch_size,
                        'num_batches': lora_microbatches_per_epoch,
                        'seed': int(cfg.training.seed) + int(self.epoch),
                        'start_batch': 0,
                        'shuffle_combined': bool(cfg.finetune.data.shuffle_combined_microbatch),
                    }

            # checkpoint
            if checkpoint_due:

                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value

                # if not cfg.policy.use_pc_color:
                #     if not os.path.exists(f'checkpoints/{self.cfg.robotwin_task.name}'):
                #         os.makedirs(f'checkpoints/{self.cfg.robotwin_task.name}')
                #     save_path = f'checkpoints/{self.cfg.robotwin_task.name}/{self.epoch + 1}.ckpt'
                # else:
                #     if not os.path.exists(f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb'):
                #         os.makedirs(f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb')
                #     save_path = f'checkpoints/{self.cfg.robotwin_task.name}_w_rgb/{self.epoch + 1}.ckpt'

                # self.save_checkpoint(save_path)
                monitor_key = cfg.checkpoint.topk.monitor_key
                if monitor_key not in metric_dict:
                    cprint(
                        f"Skipping Top-K checkpoint because '{monitor_key}' is unavailable",
                        'yellow',
                    )
                    topk_ckpt_path = None
                else:
                    try:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    except Exception as e:
                        print(f"Error in getting topk ckpt path: {e}")
                        topk_ckpt_path = None

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)


            # ========= eval end for this epoch ==========
            self.model.train()
            if self.ema_model is not None:
                self.ema_model.eval()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            del step_log

    def eval(self, mode='best'):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)

        lastest_ckpt_path = self.get_checkpoint_path(tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key)
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from {mode} checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
            # print ckpt info
            cprint(f"{self.epoch} epochs, {self.global_step} steps", 'magenta')

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.robotwin_task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        # inference_steps = cfg.policy.num_inference_steps
        all_rollout_steps = [10] # [10, 1, 4, 2, 8]
        for inference_steps in all_rollout_steps:
            eval_episodes = cfg.robotwin_task.env_runner.eval_episodes
            cprint(f"Running evaluation for {inference_steps} inference steps", 'magenta')

            horizon = policy.horizon
            n_action_steps = policy.n_action_steps
            cprint(f"Evaluating with horizon={horizon}, n_action_steps={n_action_steps}, eval_episodes={eval_episodes}, inference_steps={inference_steps}", 'magenta')

            # Create eval results directory
            eval_dir = os.path.join(self.output_dir, f'eval_results/{self.epoch}/eval_{eval_episodes}_episodes/horizon{horizon}_act{n_action_steps}/{inference_steps}')
            os.makedirs(eval_dir, exist_ok=True)

            policy.num_inference_steps = inference_steps
            runner_log = env_runner.run(policy)


            cprint(f"---------------- Eval Results --------------", 'magenta')
            metrics_dict = {}
            for key, value in runner_log.items():
                if isinstance(value, float):
                    metrics_dict[key] = value
                    cprint(f"{key}: {value:.4f}", 'magenta')
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, float):
                            metrics_dict[f"{key}/{k}"] = v
                            cprint(f"{key}/{k}: {v:.4f}", 'magenta')

            # Save metrics to JSON
            import json
            metrics_path = os.path.join(eval_dir, f'metrics_{mode}_{self.epoch}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4)

            # Save videos if they exist in runner_log
            runner_log.pop('average_success_rate', None) # Remove average_success_rate from runner_log
            video_id = 0
            task_name = runner_log['task_name']
            for k, v in runner_log.items():
                if 'video' in k:
                    if isinstance(v, np.ndarray):
                        video_dir = os.path.join(eval_dir, 'videos', task_name)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')

                        # Convert from N, C, H, W to N, H, W, C format for saving
                        v = np.transpose(v, (0, 2, 3, 1))
                        # Save video using imageio or cv2
                        import imageio
                        imageio.mimsave(video_path, v, fps=10)
                    elif hasattr(v, '_path'):  # Handle wandb.Video object
                        video_dir = os.path.join(eval_dir, 'videos', task_name)
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f'{k}_{mode}_{self.epoch}_{video_id}.mp4')
                        # Copy the video file from wandb path to our eval directory
                        shutil.copy2(v._path, video_path)
                    else:
                        cprint(f"Unknown video format for {k}", 'red')
                    video_id += 1
            cprint(f"Evaluation results saved to {eval_dir}", 'magenta')


    def get_policy_and_runner(self, cfg, checkpoint_num=3000):
        # load the latest checkpoint

        cfg = copy.deepcopy(self.cfg)
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.robotwin_task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)

        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(f'./checkpoints/{self.cfg.robotwin_task.name}/{checkpoint_num}.ckpt')
        else:
            ckpt_file = pathlib.Path(f'./checkpoints/{self.cfg.robotwin_task.name}_w_rgb/{checkpoint_num}.ckpt')

        print('ckpt file exist:', ckpt_file.is_file())

        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", 'magenta')
            self.load_checkpoint(path=ckpt_file)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model

        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def _export_lora_artifacts(self, checkpoint_path):
        if not _lora_enabled(self.cfg):
            return
        checkpoint_path = pathlib.Path(checkpoint_path)
        adapter_enabled = bool(OmegaConf.select(self.cfg, 'finetune.lora.export_adapter', default=True))
        merged_enabled = bool(OmegaConf.select(self.cfg, 'finetune.lora.export_merged', default=True))
        if adapter_enabled:
            adapter_dir = checkpoint_path.parent / f"{checkpoint_path.stem}.adapters"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(self.model.model, 'save_pretrained'):
                self.model.model.save_pretrained(adapter_dir / 'model', safe_serialization=True)
            if self.ema_model is not None and hasattr(self.ema_model.model, 'save_pretrained'):
                self.ema_model.model.save_pretrained(adapter_dir / 'ema_model', safe_serialization=True)
            metadata = {
                'base_checkpoint': str(OmegaConf.select(self.cfg, 'finetune.init_from_checkpoint', default=None)),
                'base_state_key': str(OmegaConf.select(self.cfg, 'finetune.init_state_key', default='ema_model')),
                'target_modules': list(OmegaConf.select(self.cfg, 'finetune.lora.target_modules', default=[])),
                'rank': int(OmegaConf.select(self.cfg, 'finetune.lora.rank', default=0)),
                'alpha': int(OmegaConf.select(self.cfg, 'finetune.lora.alpha', default=0)),
                'dropout': float(OmegaConf.select(self.cfg, 'finetune.lora.dropout', default=0.0)),
                'checkpoint': str(checkpoint_path),
                'optimizer_step': int(self.optimizer_step),
                'micro_step': int(self.micro_step),
            }
            with (adapter_dir / 'maniflow_adapter.json').open('w', encoding='utf-8') as file:
                json.dump(metadata, file, indent=2)
            cprint(f"[LoRA] Exported adapter artifacts to {adapter_dir}", 'cyan')

        if merged_enabled:
            merged_payload = {
                'cfg': self.cfg,
                'state_dicts': {},
                'pickles': {
                    '_output_dir': dill.dumps(self._output_dir),
                    'global_step': dill.dumps(self.global_step),
                    'optimizer_step': dill.dumps(self.optimizer_step),
                    'micro_step': dill.dumps(self.micro_step),
                    'epoch': dill.dumps(self.epoch),
                },
            }
            merged_model = copy.deepcopy(self.model)
            merged_model.to('cpu')
            if hasattr(merged_model.model, 'merge_and_unload'):
                merged_model.model = merged_model.model.merge_and_unload()
            merged_payload['state_dicts']['model'] = merged_model.state_dict()
            if any('lora_' in key for key in merged_payload['state_dicts']['model'].keys()):
                raise RuntimeError("Merged model checkpoint still contains LoRA parameters")
            if self.ema_model is not None:
                merged_ema_model = copy.deepcopy(self.ema_model)
                merged_ema_model.to('cpu')
                if hasattr(merged_ema_model.model, 'merge_and_unload'):
                    merged_ema_model.model = merged_ema_model.model.merge_and_unload()
                merged_payload['state_dicts']['ema_model'] = merged_ema_model.state_dict()
                if any('lora_' in key for key in merged_payload['state_dicts']['ema_model'].keys()):
                    raise RuntimeError("Merged EMA checkpoint still contains LoRA parameters")
            merged_path = checkpoint_path.parent / f"{checkpoint_path.stem}.merged.ckpt"
            torch.save(merged_payload, merged_path.open('wb'), pickle_module=dill)
            cprint(f"[LoRA] Exported merged dense checkpoint to {merged_path}", 'cyan')


    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        gradient_accumulate_every = int(OmegaConf.select(self.cfg, 'training.gradient_accumulate_every', default=1))
        if _lora_enabled(self.cfg) and (int(self.micro_step) % gradient_accumulate_every) != 0:
            raise RuntimeError(
                "Refusing to save a LoRA checkpoint outside a completed gradient-accumulation boundary. "
                "Partial accumulated gradients are intentionally not serialized."
            )
        print('saved in ', path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
        self.rng_state = self._capture_rng_state()

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        if not use_thread:
            self._export_lora_artifacts(path)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest', monitor_key='test_mean_score'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best':
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10 if 'loss' not in monitor_key else float('inf')
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                try:
                    # Extract score for the specified monitor_key
                    score_str = ckpt.split(f'{monitor_key}=')[1].split('.ckpt')[0]
                    score = float(score_str)

                    # Update best score based on whether we're minimizing or maximizing
                    if 'loss' in monitor_key:
                        if score < best_score:
                            best_ckpt = ckpt
                            best_score = score
                    else:
                        if score > best_score:
                            best_ckpt = ckpt
                            best_score = score
                except (IndexError, ValueError):
                    # Skip checkpoints that don't have the monitor_key
                    continue

            if best_ckpt is None:
                raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")

            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")


    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key in self.__dict__ and hasattr(self.__dict__[key], 'load_state_dict'):
                    self.__dict__[key].load_state_dict(value, **kwargs)
                else:
                    self._pending_state_dicts[key] = value
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
        self._restore_rng_state()

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
