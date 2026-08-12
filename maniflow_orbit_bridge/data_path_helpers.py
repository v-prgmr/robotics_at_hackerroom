"""Lightweight helpers shared by Orbit-to-ManiFlow data paths."""

from __future__ import annotations

import numpy as np


SOURCE_ROLE_PROGRESS_VALID = {
    "expert": True,
    "success": True,
    "hil": False,
    "correction": False,
    "failure": False,
}

COLLECTION_SOURCE_ROLES = {
    "teabags_kitting_50_v2": "expert",
    "successes": "success",
    "maniflow_hil_bounded": "correction",
    "failures": "failure",
}


def classify_source_role(role: str) -> tuple[str, bool]:
    """Validate a source role and return its canonical progress validity."""
    normalized = role.strip().lower()
    if normalized not in SOURCE_ROLE_PROGRESS_VALID:
        supported = ", ".join(sorted(SOURCE_ROLE_PROGRESS_VALID))
        raise ValueError(f"Unknown source role {role!r}; expected one of: {supported}")
    return normalized, SOURCE_ROLE_PROGRESS_VALID[normalized]


def classify_collection_role(collection: str) -> tuple[str, bool]:
    """Classify a known multi-collection source without guessing from its name."""
    try:
        role = COLLECTION_SOURCE_ROLES[collection]
    except KeyError as exc:
        supported = ", ".join(sorted(COLLECTION_SOURCE_ROLES))
        raise ValueError(
            f"Unknown Orbit collection role for {collection!r}; explicitly supported collections: {supported}"
        ) from exc
    return classify_source_role(role)


def source_episode_progress(num_frames: int, *, valid: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return exact source-frame t/(N-1) progress and its episode validity."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        progress = np.zeros(1, dtype=np.float32)
    else:
        progress = np.arange(num_frames, dtype=np.float32) / np.float32(num_frames - 1)
    progress_valid = np.full(num_frames, valid, dtype=np.bool_)
    return progress, progress_valid


def latest_real_observation_index(
    sampler_index: tuple[int, int, int, int] | np.ndarray,
    n_obs_steps: int,
) -> int:
    """Map the latest real observation in a padded prefix to replay-buffer space."""
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    buffer_start, _buffer_end, sample_start, sample_end = (int(value) for value in sampler_index)
    latest_sample_index = min(n_obs_steps, sample_end) - 1
    if latest_sample_index < sample_start:
        raise ValueError("Observation prefix does not contain a real replay-buffer observation")
    return buffer_start + latest_sample_index - sample_start


def progress_episode_masks(
    episode_ends: np.ndarray,
    progress_valid: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split supervised episodes deterministically without frame leakage."""
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    progress_valid = np.asarray(progress_valid, dtype=np.bool_)
    starts = np.concatenate(([0], episode_ends[:-1]))
    eligible = np.asarray(
        [bool(np.any(progress_valid[start:end])) for start, end in zip(starts, episode_ends, strict=True)],
        dtype=bool,
    )
    eligible_indices = np.flatnonzero(eligible)
    if val_ratio <= 0:
        return eligible, np.zeros_like(eligible)
    if len(eligible_indices) < 2:
        raise ValueError("Progress training requires at least two supervised episodes for validation")
    n_val = min(max(1, round(len(eligible_indices) * val_ratio)), len(eligible_indices) - 1)
    rng = np.random.default_rng(seed)
    val_indices = rng.choice(eligible_indices, size=n_val, replace=False)
    val = np.zeros_like(eligible)
    val[val_indices] = True
    return eligible & ~val, val
