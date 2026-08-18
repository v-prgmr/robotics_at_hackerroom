"""Evaluate held-out LeWM latent rollouts and action sensitivity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lewm_orbit_bridge.common import episode_indices_for_split, write_json
from lewm_orbit_bridge.evaluation import (
    autoregressive_rollout,
    encode_pixels,
    group_actions,
    load_evaluation,
    metric_arrays,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args(argv)
    if args.max_windows < 2:
        parser.error("--max-windows must be at least 2")
    if any(horizon <= 0 for horizon in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        parser.error("--horizons must contain unique positive integers")
    cfg, dataset, manifest, model, device, image_transform, mean, std = load_evaluation(
        args.config, args.checkpoint
    )
    history = int(cfg.history_size)
    frameskip = int(cfg.dataset.frameskip)
    max_horizon = max(args.horizons)
    test_episodes = episode_indices_for_split(dataset, manifest, "test")
    candidates = []
    for episode_idx in test_episodes:
        length = int(dataset.lengths[episode_idx])
        required_rows = (history + max_horizon) * frameskip
        for start in range(0, max(0, length - required_rows + 1), frameskip):
            candidates.append((episode_idx, start))
    if len(candidates) < 2:
        raise ValueError("Need at least two held-out windows for correct-versus-shuffled action evaluation")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.max_windows]
    if len(candidates) < 2:
        raise ValueError("Need at least two retained held-out windows")
    wrong_order = []
    for index, (episode_idx, _) in enumerate(candidates):
        choices = [other for other, (other_episode, _) in enumerate(candidates) if other_episode != episode_idx]
        if not choices:
            raise ValueError("Shuffled-action evaluation requires test windows from at least two episodes")
        wrong_order.append(choices[int(rng.integers(len(choices)))])
    records = []
    aggregates: dict[str, list[float]] = defaultdict(list)

    for window_idx, (episode_idx, start) in enumerate(candidates):
        end = start + (history + max_horizon) * frameskip
        sample = dataset._load_slice(episode_idx, start, end)
        wrong_ep, wrong_start = candidates[int(wrong_order[window_idx])]
        wrong = dataset._load_slice(wrong_ep, wrong_start, wrong_start + (history + max_horizon) * frameskip)
        real_embeddings = encode_pixels(model, sample["pixels"][::frameskip], image_transform, device)
        initial = real_embeddings[:history]
        correct_actions = group_actions(sample["action"], frameskip, mean, std)
        wrong_actions = group_actions(wrong["action"], frameskip, mean, std)
        predicted = autoregressive_rollout(model, initial, correct_actions, max_horizon)
        shuffled = autoregressive_rollout(model, initial, wrong_actions, max_horizon)
        target = real_embeddings[history : history + max_horizon]
        persistence = initial[-1:].expand_as(target)
        correct_mse, correct_cos = metric_arrays(predicted, target)
        shuffled_mse, shuffled_cos = metric_arrays(shuffled, target)
        persistence_mse, persistence_cos = metric_arrays(persistence, target)
        for horizon in args.horizons:
            idx = horizon - 1
            row = {
                "episode_index": episode_idx,
                "start_row": start,
                "wrong_episode_index": wrong_ep,
                "wrong_start_row": wrong_start,
                "horizon": horizon,
                "correct_mse": float(correct_mse[idx]),
                "correct_cosine": float(correct_cos[idx]),
                "shuffled_mse": float(shuffled_mse[idx]),
                "shuffled_cosine": float(shuffled_cos[idx]),
                "persistence_mse": float(persistence_mse[idx]),
                "persistence_cosine": float(persistence_cos[idx]),
                "correct_beats_shuffled": bool(correct_mse[idx] < shuffled_mse[idx]),
                "correct_beats_persistence": bool(correct_mse[idx] < persistence_mse[idx]),
            }
            records.append(row)
            for key in ("correct_mse", "shuffled_mse", "persistence_mse", "correct_cosine"):
                aggregates[f"{horizon}/{key}"].append(float(row[key]))

    summary = {"split": "test", "number_of_windows": len(candidates), "horizons": {}}
    for horizon in args.horizons:
        rows = [row for row in records if row["horizon"] == horizon]
        summary["horizons"][str(horizon)] = {
            "correct_mse_mean": float(np.mean([row["correct_mse"] for row in rows])),
            "shuffled_mse_mean": float(np.mean([row["shuffled_mse"] for row in rows])),
            "persistence_mse_mean": float(np.mean([row["persistence_mse"] for row in rows])),
            "correct_cosine_mean": float(np.mean([row["correct_cosine"] for row in rows])),
            "correct_beats_shuffled_rate": float(np.mean([row["correct_beats_shuffled"] for row in rows])),
            "correct_beats_persistence_rate": float(np.mean([row["correct_beats_persistence"] for row in rows])),
            "action_sensitivity_margin": float(np.mean([row["shuffled_mse"] - row["correct_mse"] for row in rows])),
        }
    output = Path(str(cfg.output_dir)).expanduser().resolve() / "latent_prediction"
    write_json(output / "summary.json", summary)
    write_json(output / "windows.json", records)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for key, label in (("correct_mse", "LeWM correct actions"), ("shuffled_mse", "LeWM shuffled actions"), ("persistence_mse", "Persistence")):
        axis.plot(args.horizons, [summary["horizons"][str(h)][f"{key}_mean"] for h in args.horizons], marker="o", label=label)
    axis.set(xlabel="Rollout horizon (world-model transitions)", ylabel="Latent MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "error_vs_horizon.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
