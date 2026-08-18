"""Explore successful-goal structure in held-out LeWM latent trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lewm_orbit_bridge.common import episode_indices_for_split, write_json
from lewm_orbit_bridge.evaluation import encode_pixels, load_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=30)
    args = parser.parse_args(argv)
    cfg, dataset, manifest, model, device, image_transform, _, _ = load_evaluation(
        args.config, args.checkpoint
    )
    metadata = dataset.get_episode_data()
    train_indices = episode_indices_for_split(dataset, manifest, "train")
    test_indices = episode_indices_for_split(dataset, manifest, "test")
    goal_indices = [idx for idx in train_indices if bool(metadata["goal_eligible"][idx])]
    if not goal_indices:
        raise ValueError("Training split contains no genuinely successful/expert goal episodes")
    goals = []
    for episode_idx in goal_indices:
        episode = dataset.load_episode(episode_idx)
        goals.append(encode_pixels(model, episode["pixels"][-1:], image_transform, device)[0])
    import torch

    goal_embeddings = torch.stack(goals)
    goal_mean = goal_embeddings.mean(dim=0)
    records = []
    for episode_idx in test_indices:
        episode = dataset.load_episode(episode_idx)
        indices = list(range(0, len(episode["pixels"]), args.sample_stride))
        if indices[-1] != len(episode["pixels"]) - 1:
            indices.append(len(episode["pixels"]) - 1)
        embeddings = encode_pixels(model, episode["pixels"][indices], image_transform, device)
        nearest = (embeddings[:, None] - goal_embeddings[None]).pow(2).mean(dim=-1).min(dim=1).values
        mean_cost = (embeddings - goal_mean).pow(2).mean(dim=-1)
        progress = np.asarray(indices, dtype=np.float64) / max(1, len(episode["pixels"]) - 1)
        for position, nearest_cost, centroid_cost in zip(progress, nearest.cpu(), mean_cost.cpu(), strict=True):
            records.append(
                {
                    "episode_index": episode_idx,
                    "episode_uid": metadata["episode_uid"][episode_idx],
                    "collection_type": metadata["collection_type"][episode_idx],
                    "terminal_success_known": bool(metadata["terminal_success_known"][episode_idx]),
                    "terminal_success": bool(metadata["terminal_success"][episode_idx]),
                    "progress": float(position),
                    "nearest_goal_cost": float(nearest_cost),
                    "mean_goal_cost": float(centroid_cost),
                }
            )
    episode_summaries = []
    for episode_idx in test_indices:
        rows = [row for row in records if row["episode_index"] == episode_idx]
        episode_summaries.append(
            {
                "episode_index": episode_idx,
                "episode_uid": rows[0]["episode_uid"],
                "collection_type": rows[0]["collection_type"],
                "nearest_goal_delta": rows[-1]["nearest_goal_cost"] - rows[0]["nearest_goal_cost"],
                "mean_goal_delta": rows[-1]["mean_goal_cost"] - rows[0]["mean_goal_cost"],
            }
        )
    summary = {
        "goal_source_split": "train",
        "evaluation_split": "test",
        "number_of_goal_embeddings": len(goals),
        "number_of_test_episodes": len(test_indices),
        "fraction_approaching_nearest_goal": float(np.mean([row["nearest_goal_delta"] < 0 for row in episode_summaries])),
        "fraction_approaching_mean_goal": float(np.mean([row["mean_goal_delta"] < 0 for row in episode_summaries])),
        "caveat": "Exploratory latent geometry; Euclidean distance is not assumed to be a calibrated value function.",
    }
    output = Path(str(cfg.output_dir)).expanduser().resolve() / "goal_structure"
    write_json(output / "summary.json", summary)
    write_json(output / "trajectory_costs.json", records)
    write_json(output / "episode_summaries.json", episode_summaries)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for episode_idx in test_indices:
        rows = [row for row in records if row["episode_index"] == episode_idx]
        axes[0].plot([r["progress"] for r in rows], [r["nearest_goal_cost"] for r in rows], alpha=0.45)
        axes[1].plot([r["progress"] for r in rows], [r["mean_goal_cost"] for r in rows], alpha=0.45)
    axes[0].set_title("Nearest successful goal")
    axes[1].set_title("Mean successful goal")
    for axis in axes:
        axis.set(xlabel="Episode progress", ylabel="Latent MSE")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "goal_cost_trajectories.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
