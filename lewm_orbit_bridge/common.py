"""Shared dataset, split, normalization, and model helpers for LeWM."""

from __future__ import annotations

import importlib
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_LEWM_SHA = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
SPLIT_NAMES = ("train", "val", "test")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_collection_type(path: Path) -> str:
    name = str(path).lower()
    if "hil" in name or "rac" in name or "correction" in name:
        return "hil/rac_correction"
    if "fail" in name:
        return "policy_failure"
    if "success" in name or "successes" in name:
        return "policy_success"
    return "expert"


def parse_dataset_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed = []
    valid = {"expert", "policy_success", "policy_failure", "hil/rac_correction"}
    for spec in specs:
        if "=" in spec:
            kind, raw_path = spec.split("=", 1)
            if kind not in valid:
                raise ValueError(f"Unknown collection type {kind!r}; expected one of {sorted(valid)}")
            path = Path(raw_path).expanduser().resolve()
        else:
            path = Path(spec).expanduser().resolve()
            kind = infer_collection_type(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {path}")
        parsed.append((kind, path))
    return parsed


def stratified_episode_split(
    episodes: list[dict[str, Any]], seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[str]]:
    """Split stable episode UIDs by collection type, reproducibly."""

    if not np.isclose(sum(ratios), 1.0) or any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative and sum to one")
    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode["collection_type"])].append(str(episode["episode_uid"]))

    result = {name: [] for name in SPLIT_NAMES}
    rng = np.random.default_rng(seed)
    for kind in sorted(grouped):
        values = np.asarray(sorted(grouped[kind]), dtype=object)
        rng.shuffle(values)
        n = len(values)
        raw = np.asarray(ratios) * n
        counts = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
            counts[idx] += 1
        if n >= 3:
            for required in (1, 2):
                if ratios[required] > 0 and counts[required] == 0:
                    donor = int(np.argmax(counts))
                    counts[donor] -= 1
                    counts[required] += 1
        boundaries = np.cumsum(counts)
        chunks = np.split(values, boundaries[:-1])
        for name, chunk in zip(SPLIT_NAMES, chunks, strict=True):
            result[name].extend(str(v) for v in chunk)
    for values in result.values():
        values.sort()
    return result


def episode_indices_for_split(dataset: Any, manifest: dict[str, Any], split: str) -> list[int]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    metadata = dataset.get_episode_data()
    uids = metadata.get("episode_uid")
    if uids is None:
        raise ValueError("Lance dataset lacks episode_uid metadata")
    if len(uids) != len(set(uids)):
        raise ValueError("Lance dataset contains duplicate episode_uid values")
    split_lists = manifest.get("splits", {})
    if set(split_lists) != set(SPLIT_NAMES):
        raise ValueError(f"Split manifest must contain exactly {SPLIT_NAMES}")
    if any(len(values) != len(set(values)) for values in split_lists.values()):
        raise ValueError("Split manifest contains duplicate episode UIDs")
    split_sets = {name: set(values) for name, values in split_lists.items()}
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Train, validation, and test episode splits overlap")
    if set().union(*split_sets.values()) != set(uids):
        raise ValueError("Split manifest is not a complete partition of the Lance episodes")
    wanted = split_sets[split]
    indices = [idx for idx, uid in enumerate(uids) if uid in wanted]
    found = {uids[idx] for idx in indices}
    missing = wanted - found
    if missing:
        raise ValueError(f"Split manifest references missing episodes: {sorted(missing)[:5]}")
    return indices


def clip_indices_for_episodes(dataset: Any, episode_indices: Iterable[int]) -> list[int]:
    allowed = set(int(v) for v in episode_indices)
    return [idx for idx, (episode_idx, _) in enumerate(dataset.clip_indices) if episode_idx in allowed]


def fit_action_normalization(dataset: Any, episode_indices: Iterable[int]) -> dict[str, Any]:
    chunks = []
    for episode_idx in episode_indices:
        action = dataset.load_episode(int(episode_idx))["action"].detach().cpu().numpy()
        chunks.append(action.reshape(-1, action.shape[-1]))
    if not chunks:
        raise ValueError("Cannot fit action normalization on an empty training split")
    values = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training actions contain NaN or infinity")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return {
        "method": "zscore",
        "fit_split": "train",
        "count": int(len(values)),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def prepare_lewm_imports(lewm_root: Path) -> None:
    root = str(lewm_root.expanduser().resolve())
    if not (Path(root) / "jepa.py").is_file():
        raise FileNotFoundError(f"LeWM checkout not found at {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    for module in ("jepa", "module", "utils"):
        importlib.import_module(module)


def load_lewm_model(checkpoint: Path, lewm_root: Path, config_path: Path | None = None):
    """Load an upstream object checkpoint or instantiate from a weights checkpoint."""

    import hydra
    import torch
    from omegaconf import OmegaConf

    prepare_lewm_imports(lewm_root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.nn.Module):
        return payload
    if hasattr(payload, "model") and isinstance(payload.model, torch.nn.Module):
        return payload.model
    if config_path is None:
        raise ValueError("A resolved config is required for a weights-only checkpoint")
    cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(cfg.model)
    state = payload.get("state_dict", payload)
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model
