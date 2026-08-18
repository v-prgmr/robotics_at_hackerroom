"""Train stock LeWM with episode-disjoint splits and train-only normalization."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from functools import partial
from pathlib import Path

import numpy as np

from lewm_orbit_bridge.common import (
    clip_indices_for_episodes,
    episode_indices_for_split,
    fit_action_normalization,
    file_sha256,
    prepare_lewm_imports,
    read_json,
    write_json,
)


class NormalizeAction:
    """Picklable train-fitted action transform for DataLoader workers."""

    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, batch):
        batch["action"] = ((batch["action"] - self.mean) / self.std).float()
        return batch


def _git_sha(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "tiny-overfit", "full"), default="full")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args, overrides = parser.parse_known_args(argv)

    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from omegaconf import OmegaConf, open_dict

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    if args.batch_size:
        cfg.loader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.loader.num_workers = args.num_workers
    if args.mode == "smoke":
        cfg.trainer.max_epochs = 1
        cfg.trainer.max_steps = args.max_steps or 50
    elif args.mode == "tiny-overfit":
        cfg.trainer.max_epochs = 1000
        cfg.trainer.max_steps = args.max_steps or 400
    elif args.max_steps is not None:
        cfg.trainer.max_steps = args.max_steps

    lewm_root = Path(str(cfg.lewm_root)).expanduser().resolve()
    prepare_lewm_imports(lewm_root)
    from module import SIGReg
    from utils import get_img_preprocessor

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = read_json(Path(str(cfg.split_manifest)).expanduser())
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset = swm.data.load_dataset(str(Path(str(cfg.dataset_path)).expanduser()), transform=None, **dataset_cfg)
    dataset_manifest_path = Path(str(cfg.dataset_path)).expanduser().with_suffix(".manifest.json")
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"Dataset conversion manifest is required: {dataset_manifest_path}")
    shutil.copy2(dataset_manifest_path, output_dir / "dataset_statistics.json")
    train_episodes = episode_indices_for_split(dataset, split_manifest, "train")
    val_episodes = episode_indices_for_split(dataset, split_manifest, "val")
    if args.mode == "tiny-overfit":
        train_episodes = train_episodes[:2]
        val_episodes = train_episodes

    normalization = fit_action_normalization(dataset, train_episodes)
    write_json(output_dir / "action_normalization.json", normalization)
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)

    transforms = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size=int(cfg.img_size)), NormalizeAction(mean, std)
    )
    dataset.transform = transforms
    train_clip_indices = clip_indices_for_episodes(dataset, train_episodes)
    val_clip_indices = clip_indices_for_episodes(dataset, val_episodes)
    if not train_clip_indices or not val_clip_indices:
        raise ValueError("Train and validation splits must each contain at least one valid clip")
    train_set = torch.utils.data.Subset(dataset, train_clip_indices)
    val_set = torch.utils.data.Subset(dataset, val_clip_indices)

    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if int(loader_cfg["num_workers"]) == 0:
        loader_cfg.pop("persistent_workers", None)
        loader_cfg.pop("prefetch_factor", None)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = torch.utils.data.DataLoader(
        train_set, **loader_cfg, shuffle=True, drop_last=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(val_set, **loader_cfg, shuffle=False, drop_last=False)

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = int(cfg.dataset.frameskip) * dataset.get_dim("action")

    short_run_losses = []

    def forward(self, batch, stage):
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.model.encode(batch)
        embeddings = output["emb"]
        context = embeddings[:, : int(cfg.history_size)]
        context_actions = output["act_emb"][:, : int(cfg.history_size)]
        target = embeddings[:, int(cfg.num_preds) :]
        prediction = self.model.predict(context, context_actions)
        output["pred_loss"] = (prediction - target).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(embeddings.transpose(0, 1))
        output["loss"] = output["pred_loss"] + float(cfg.loss.sigreg.weight) * output["sigreg_loss"]
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError(f"Non-finite {stage} loss")
        if stage == "train" and args.mode in {"smoke", "tiny-overfit"}:
            short_run_losses.append(float(output["pred_loss"].detach().cpu()))
        self.log_dict({f"{stage}/{key}": value for key, value in output.items() if "loss" in key}, on_step=True, on_epoch=True)
        return output

    model = hydra.utils.instantiate(cfg.model)
    optimizer = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**dict(cfg.loss.sigreg.kwargs)),
        forward=forward,
        optim=optimizer,
    )

    loggers = [CSVLogger(output_dir, name="training_curves")]
    if bool(cfg.wandb.enabled):
        loggers.append(WandbLogger(project=str(cfg.wandb.project), name=str(cfg.experiment_name)))
    checkpoint = ModelCheckpoint(dirpath=output_dir / "checkpoints", save_last=True, save_top_k=1, monitor="val/pred_loss_epoch", mode="min")
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=[checkpoint],
        logger=loggers,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
    )
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    write_json(
        output_dir / "git_versions.json",
        {"orbit": _git_sha(Path(__file__).resolve().parents[1]), "lewm": _git_sha(lewm_root)},
    )
    write_json(output_dir / "splits.json", split_manifest)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("Training completed without a validation-selected checkpoint")
    best_payload = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    module.load_state_dict(best_payload["state_dict"], strict=True)
    torch.save(module.model, output_dir / "lewm_object.ckpt")
    torch.save(module.model.state_dict(), output_dir / "lewm_weights.pt")
    if short_run_losses:
        initial_prediction_loss = float(np.mean(short_run_losses[: min(20, len(short_run_losses))]))
        final_prediction_loss = float(np.mean(short_run_losses[-min(20, len(short_run_losses)) :]))
    else:
        initial_prediction_loss = None
        final_prediction_loss = None
    if (
        args.mode == "tiny-overfit"
        and initial_prediction_loss is not None
        and final_prediction_loss is not None
        and final_prediction_loss >= 0.8 * initial_prediction_loss
    ):
        raise RuntimeError(
            "Tiny-subset overfit failed to reduce prediction loss by at least 20%: "
            f"initial={initial_prediction_loss:.6g}, final={final_prediction_loss:.6g}"
        )
    summary = {
        "mode": args.mode,
        "optimizer_steps": int(trainer.global_step),
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "train_clips": len(train_clip_indices),
        "val_clips": len(val_clip_indices),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "checkpoint_saved": (output_dir / "lewm_object.ckpt").exists(),
        "selected_checkpoint": checkpoint.best_model_path,
        "selected_val_prediction_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        "initial_prediction_loss": initial_prediction_loss,
        "final_prediction_loss": final_prediction_loss,
    }
    write_json(output_dir / f"{args.mode}_summary.json", summary)
    write_json(
        output_dir / "run_manifest.json",
        {
            "resolved_config_sha256": file_sha256(output_dir / "resolved_config.yaml"),
            "splits_sha256": file_sha256(output_dir / "splits.json"),
            "action_normalization_sha256": file_sha256(output_dir / "action_normalization.json"),
            "checkpoint_sha256": file_sha256(output_dir / "lewm_object.ckpt"),
            "dataset_statistics_sha256": file_sha256(output_dir / "dataset_statistics.json"),
            "selected_checkpoint": checkpoint.best_model_path,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
