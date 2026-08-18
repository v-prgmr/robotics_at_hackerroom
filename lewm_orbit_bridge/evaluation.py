"""Shared LeWM evaluation setup and latent rollout operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lewm_orbit_bridge.common import (
    episode_indices_for_split,
    file_sha256,
    load_lewm_model,
    read_json,
)


def load_evaluation(config_path: Path, checkpoint: Path):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(config_path)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    run_manifest = read_json(output_dir / "run_manifest.json")
    expected_hashes = {
        "resolved_config_sha256": output_dir / "resolved_config.yaml",
        "splits_sha256": output_dir / "splits.json",
        "action_normalization_sha256": output_dir / "action_normalization.json",
        "dataset_statistics_sha256": output_dir / "dataset_statistics.json",
    }
    for key, path in expected_hashes.items():
        if file_sha256(path) != run_manifest[key]:
            raise ValueError(f"Run artifact hash mismatch: {path}")
    if file_sha256(checkpoint) != run_manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the configured LeWM run manifest")
    normalization = read_json(output_dir / "action_normalization.json")
    split_manifest = read_json(output_dir / "splits.json")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg["num_steps"] = 1
    dataset_cfg["frameskip"] = 1
    dataset_cfg["keys_to_load"] = ["pixels", "action"]
    dataset_cfg.pop("keys_to_cache", None)
    dataset = swm.data.load_dataset(str(Path(str(cfg.dataset_path)).expanduser()), **dataset_cfg)
    model = load_lewm_model(checkpoint, Path(str(cfg.lewm_root)), output_dir / "resolved_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    from utils import get_img_preprocessor

    image_transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size=int(cfg.img_size))
    )
    mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalization["std"], dtype=torch.float32, device=device)
    return cfg, dataset, split_manifest, model, device, image_transform, mean, std


def encode_pixels(model: Any, pixels, image_transform: Any, device):
    import torch

    transformed = image_transform({"pixels": pixels.clone()})["pixels"].to(device)
    with torch.inference_mode():
        return model.encode({"pixels": transformed.unsqueeze(0)})["emb"][0]


def group_actions(actions, frameskip: int, mean, std):
    normalized = (actions.to(mean.device) - mean) / std
    usable = (len(normalized) // frameskip) * frameskip
    return normalized[:usable].reshape(-1, frameskip * normalized.shape[-1])


def autoregressive_rollout(model: Any, initial_embeddings, grouped_actions, horizon: int):
    """Match upstream JEPA.rollout while returning only predicted futures."""

    embeddings = initial_embeddings.unsqueeze(0)
    actions = grouped_actions[: embeddings.shape[1]].unsqueeze(0)
    predictions = []
    with __import__("torch").inference_mode():
        for step in range(horizon):
            action_embeddings = model.action_encoder(actions[:, -embeddings.shape[1] :])
            prediction = model.predict(embeddings[:, -initial_embeddings.shape[0] :], action_embeddings[:, -initial_embeddings.shape[0] :])[:, -1:]
            predictions.append(prediction[:, 0])
            embeddings = __import__("torch").cat((embeddings, prediction), dim=1)
            next_index = initial_embeddings.shape[0] + step
            if next_index < len(grouped_actions):
                actions = __import__("torch").cat((actions, grouped_actions[next_index : next_index + 1].unsqueeze(0)), dim=1)
    return __import__("torch").cat(predictions, dim=0)


def metric_arrays(predicted, target) -> tuple[np.ndarray, np.ndarray]:
    import torch

    mse = (predicted - target).pow(2).mean(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    return mse.detach().cpu().numpy(), cosine.detach().cpu().numpy()
