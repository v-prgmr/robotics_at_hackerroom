"""Install Orbit ManiFlow bridge files into a ManiFlow checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Target exists: {target}. Pass --overwrite to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Installed {target}")


def _patch_text(text: str, *, target: Path, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not patch {target}; expected source block was not found.")
    return text.replace(old, new, 1)


def _patch_robotwin_workspace(package_dir: Path) -> None:
    target = package_dir / "workspace/train_maniflow_robotwin_workspace.py"
    text = target.read_text()

    old_resume = """        # resume training\n        if cfg.training.resume:\n            lastest_ckpt_path = self.get_checkpoint_path()\n            if lastest_ckpt_path.is_file():\n                print(f\"Resuming from checkpoint {lastest_ckpt_path}\")\n                self.load_checkpoint(path=lastest_ckpt_path)\n"""
    new_resume = """        # resume training\n        resume_from_checkpoint = OmegaConf.select(cfg, 'training.resume_from_checkpoint')\n        should_resume = bool(cfg.training.resume) or bool(resume_from_checkpoint)\n        if should_resume:\n            if resume_from_checkpoint:\n                lastest_ckpt_path = pathlib.Path(str(resume_from_checkpoint)).expanduser()\n                if not lastest_ckpt_path.is_file():\n                    raise FileNotFoundError(f\"Resume checkpoint does not exist: {lastest_ckpt_path}\")\n            else:\n                lastest_ckpt_path = self.get_checkpoint_path()\n            if lastest_ckpt_path.is_file():\n                print(f\"Resuming from checkpoint {lastest_ckpt_path}\")\n                self.load_checkpoint(path=lastest_ckpt_path)\n                print(f\"Loaded checkpoint epoch={self.epoch}, global_step={self.global_step}\")\n                if bool(OmegaConf.select(cfg, 'training.advance_epoch_on_resume', default=False)):\n                    self.epoch += 1\n                    self.global_step += 1\n                    print(f\"Advanced resume position to epoch={self.epoch}, global_step={self.global_step}\")\n            else:\n                print(f\"No checkpoint found at {lastest_ckpt_path}; starting from scratch\")\n\n        target_epoch = OmegaConf.select(cfg, 'training.target_epoch')\n        if target_epoch is not None:\n            target_epoch = int(target_epoch)\n            if self.epoch > target_epoch:\n                print(f\"Current epoch {self.epoch} is greater than target_epoch {target_epoch}; nothing to train\")\n                return\n"""

    old_scheduler = """        # configure lr scheduler\n        lr_scheduler = get_scheduler(\n            cfg.training.lr_scheduler,\n            optimizer=self.optimizer,\n            num_warmup_steps=cfg.training.lr_warmup_steps,\n            num_training_steps=(\n                len(train_dataloader) * cfg.training.num_epochs) \\\n                    // cfg.training.gradient_accumulate_every,\n            # pytorch assumes stepping LRScheduler every epoch\n            # however huggingface diffusers steps it every batch\n            last_epoch=self.global_step-1\n        )\n"""
    new_scheduler = """        # configure lr scheduler\n        lr_num_epochs = OmegaConf.select(cfg, 'training.lr_num_epochs')\n        if lr_num_epochs is None:\n            lr_num_epochs = cfg.training.num_epochs\n        lr_num_epochs = int(lr_num_epochs)\n        gradient_accumulate_every = int(cfg.training.gradient_accumulate_every)\n        scheduler_step = (int(self.global_step) + gradient_accumulate_every - 1) // gradient_accumulate_every\n        lr_scheduler = get_scheduler(\n            cfg.training.lr_scheduler,\n            optimizer=self.optimizer,\n            num_warmup_steps=cfg.training.lr_warmup_steps,\n            num_training_steps=(\n                len(train_dataloader) * lr_num_epochs) \\\n                    // cfg.training.gradient_accumulate_every,\n            # pytorch assumes stepping LRScheduler every epoch\n            # however huggingface diffusers steps it every batch\n            last_epoch=scheduler_step-1\n        )\n        print(\n            f\"LR scheduler resume: lr_num_epochs={lr_num_epochs}, \"\n            f\"scheduler_step={scheduler_step}, lr={lr_scheduler.get_last_lr()[0]:.8g}\"\n        )\n"""

    old_loop = """        # training loop\n        log_path = os.path.join(self.output_dir, 'logs.json.txt')\n        for local_epoch_idx in range(cfg.training.num_epochs):\n"""
    new_loop = """        # training loop\n        log_path = os.path.join(self.output_dir, 'logs.json.txt')\n        if target_epoch is None:\n            epochs_to_run = cfg.training.num_epochs\n        else:\n            epochs_to_run = target_epoch - self.epoch + 1\n        print(f\"Training for {epochs_to_run} epoch(s), starting at epoch {self.epoch}\")\n        for local_epoch_idx in range(epochs_to_run):\n"""

    text = _patch_text(text, target=target, old=old_resume, new=new_resume)
    text = _patch_text(text, target=target, old=old_scheduler, new=new_scheduler)
    text = _patch_text(text, target=target, old=old_loop, new=new_loop)
    target.write_text(text)
    print(f"Patched {target}")


def install(maniflow_dir: Path, *, overwrite: bool = False) -> None:
    source_dir = Path(__file__).resolve().parent
    maniflow_dir = maniflow_dir.expanduser().resolve()
    package_dir = maniflow_dir / "maniflow"
    if not package_dir.exists():
        raise FileNotFoundError(f"Expected ManiFlow package directory at {package_dir}")

    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Installed {init_file}")

    files = [
        (
            source_dir / "maniflow_dataset/orbit_image_dataset.py",
            package_dir / "dataset/orbit_image_dataset.py",
        ),
        (
            source_dir / "maniflow_config/maniflow_image_orbit.yaml",
            package_dir / "config/maniflow_image_orbit.yaml",
        ),
        (
            source_dir / "maniflow_config/robotwin_task/orbit_so100_image.yaml",
            package_dir / "config/robotwin_task/orbit_so100_image.yaml",
        ),
        (
            source_dir / "maniflow_workspace/train_maniflow_orbit_workspace.py",
            package_dir / "workspace/train_maniflow_orbit_workspace.py",
        ),
    ]
    for source, target in files:
        _copy_file(source, target, overwrite=overwrite)
    _patch_robotwin_workspace(package_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maniflow-dir", type=Path, required=True, help="Path to allenai/maniflow checkout.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing installed bridge files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    install(args.maniflow_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
