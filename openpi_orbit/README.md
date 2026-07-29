# Orbit OpenPI / pi0.5 Integration

This folder contains the Orbit-specific OpenPI adapters and a small train wrapper for fine-tuning pi0.5 on Orbit-exported LeRobot datasets.

The existing Orbit export already produces the LeRobot fields the adapter expects:

- `observation.state`
- `action`
- `task`
- `observation.images.overhead`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

## Workflow

1. Export Orbit data to LeRobot format from the repository root:

```bash
uv run bimanual-export-lerobot \
    --input-dir ./data/bimanual \
    --output-dir ./dataset/orbit_so100_lerobot \
    --repo-id local/orbit_so100 \
    --fps 60 \
    --video-codec h264 \
    --encoder-threads 1
```

2. Clone and install OpenPI in a separate checkout.

3. Install the Orbit adapters into that OpenPI checkout and run norm-stat computation plus training:

```bash
python openpi_orbit/train_pi05_orbit.py \
    --openpi-dir /path/to/openpi \
    --dataset-root ./dataset/orbit_so100_lerobot \
    --repo-id local/orbit_so100 \
    --exp-name orbit_smoke \
    --steps 1000 \
    --batch-size 8 \
    --overwrite
```

The train wrapper installs the adapter files into OpenPI, converts Orbit's LeRobot v3 export into the older LeRobot v2.1 layout that OpenPI currently expects, symlinks that compatible dataset into the Hugging Face LeRobot cache path, computes normalization stats, and launches `scripts/train.py` with the `pi05_orbit_so100` config.

The generated compatibility export is written next to the v3 export, for example:

```text
dataset/orbit_so100_lerobot_openpi_v21/
```

## Files

- `orbit_policy.py`: OpenPI input/output transforms for Orbit bimanual SO-100 data.
- `orbit_data_config.py`: OpenPI `DataConfigFactory` and `TrainConfig` registration for pi0.5.
- `install_into_openpi.py`: Copies adapters into OpenPI and registers the config.
- `train_pi05_orbit.py`: End-to-end helper for adapter installation, norm stats, and training.

## Action Semantics

Orbit exports absolute commanded joint targets in this order:

```text
left shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper,
right shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The adapter uses a delta-action transform for the five non-gripper joints on each arm and leaves both gripper dimensions absolute.
