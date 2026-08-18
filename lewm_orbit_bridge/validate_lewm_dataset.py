"""Strictly validate an Orbit-derived LeWM Lance dataset and render transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lewm_orbit_bridge.common import write_json


def validate_dataset(path: Path) -> tuple[dict, list[str]]:
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        str(path), num_steps=1, frameskip=1, keys_to_load=[
            "pixels",
            "action",
            "source_timestep_index",
            "source_camera_frame_index",
            "source_video_frame_index",
        ]
    )
    metadata = dataset.get_episode_data()
    errors: list[str] = []
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    actions = []
    rgb_shapes: Counter[tuple[int, ...]] = Counter()
    duplicate_fractions = []
    for episode_idx, expected_length in enumerate(lengths):
        episode = dataset.load_episode(episode_idx)
        pixels = episode["pixels"].detach().cpu().numpy()
        action = episode["action"].detach().cpu().numpy()
        timestep = episode["source_timestep_index"].detach().cpu().numpy().reshape(-1)
        camera = episode["source_camera_frame_index"].detach().cpu().numpy().reshape(-1)
        video = episode["source_video_frame_index"].detach().cpu().numpy().reshape(-1)
        if len(pixels) != len(action) or len(pixels) != int(expected_length):
            errors.append(f"episode {episode_idx}: pixel/action/boundary length mismatch")
        if pixels.dtype != np.uint8:
            errors.append(f"episode {episode_idx}: RGB dtype is {pixels.dtype}, expected uint8")
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            errors.append(f"episode {episode_idx}: RGB tensor shape {pixels.shape} is not TCHW RGB")
        else:
            rgb_shapes[tuple(pixels.shape[1:])] += 1
        if not np.isfinite(action).all():
            errors.append(f"episode {episode_idx}: actions contain NaN or infinity")
        if len(timestep) > 1 and np.any(np.diff(timestep) <= 0):
            errors.append(f"episode {episode_idx}: timestep indices are not strictly increasing")
        for name, values in (("camera", camera), ("video", video)):
            if len(values) > 1 and np.any(np.diff(values) < 0):
                errors.append(f"episode {episode_idx}: overhead {name} indices decrease")
        duplicate_fractions.append(1.0 - len(np.unique(video)) / len(video))
        actions.append(action)
    if len(rgb_shapes) != 1:
        errors.append(f"RGB shape differs between episodes: {dict(rgb_shapes)}")
    action_dims = {array.shape[-1] for array in actions}
    if len(action_dims) != 1:
        errors.append(f"Action dimension differs between episodes: {sorted(action_dims)}")
    all_actions = np.concatenate(actions) if actions else np.empty((0, 0), dtype=np.float32)
    report = {
        "number_of_episodes": int(len(lengths)),
        "number_of_timesteps": int(lengths.sum()),
        "usable_raw_transitions": int(sum(max(0, int(value) - 1) for value in lengths)),
        "trajectory_length": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "rgb_shapes_chw": {str(key): value for key, value in rgb_shapes.items()},
        "action_dimension": int(all_actions.shape[-1]) if all_actions.size else 0,
        "actions_finite": bool(np.isfinite(all_actions).all()),
        "action_min": all_actions.min(axis=0).tolist() if all_actions.size else [],
        "action_max": all_actions.max(axis=0).tolist() if all_actions.size else [],
        "action_mean": all_actions.mean(axis=0).tolist() if all_actions.size else [],
        "action_std": all_actions.std(axis=0).tolist() if all_actions.size else [],
        "collection_type_counts": dict(Counter(metadata.get("collection_type", []))),
        "split_counts": dict(Counter(metadata.get("split", []))),
        "mean_duplicate_video_frame_fraction": float(np.mean(duplicate_fractions)) if duplicate_fractions else 0.0,
        "errors": errors,
    }
    return report, errors


def render_samples(path: Path, output: Path, count: int, seed: int) -> None:
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(str(path), num_steps=2, frameskip=1, keys_to_load=["pixels", "action"])
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3.4 * len(selected)), squeeze=False)
    for row, clip_idx in enumerate(selected):
        sample = dataset[int(clip_idx)]
        images = sample["pixels"].permute(0, 2, 3, 1).cpu().numpy()
        action = sample["action"][0].cpu().numpy()
        axes[row, 0].imshow(images[0])
        axes[row, 0].set_title("observation_t")
        axes[row, 1].axis("off")
        axes[row, 1].text(0.02, 0.5, np.array2string(action, precision=3), family="monospace", wrap=True)
        axes[row, 1].set_title("action_t")
        axes[row, 2].imshow(images[1])
        axes[row, 2].set_title("observation_t+1")
        axes[row, 0].axis("off")
        axes[row, 2].axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--visual-sample", type=Path, default=Path("outputs/lewm/dataset_transitions.png"))
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args(argv)
    report, errors = validate_dataset(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        write_json(args.report, report)
    if not errors:
        render_samples(args.dataset, args.visual_sample, args.sample_count, args.seed)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
