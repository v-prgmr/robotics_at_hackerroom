# Orbit ManiFlow Bridge

Utilities for training AllenAI ManiFlow on Orbit bimanual SO-100 datasets with a multi-camera, language-conditioned 2D policy.

This bridge is intentionally separate from the Orbit package. Orbit records/exports data; ManiFlow trains the policy in its own environment.

## Files

- `convert_orbit_lerobot_to_maniflow.py`: converts an Orbit LeRobot v3 export into a ManiFlow `.zarr` replay buffer.
- `install_into_maniflow.py`: copies Orbit-specific dataset/config/workspace files into a ManiFlow checkout.
- `maniflow_dataset/orbit_image_dataset.py`: ManiFlow dataset class for `overhead`, `left_wrist`, `right_wrist`, `state`, `action`, and language task strings.
- `maniflow_workspace/train_maniflow_orbit_workspace.py`: image-only trainer wrapper that avoids ManiFlow's unused PyTorch3D pointcloud import.
- `maniflow_config/maniflow_image_orbit.yaml`: main ManiFlow Hydra config for language-conditioned 2D training.
- `maniflow_config/robotwin_task/orbit_so100_image.yaml`: Orbit task/dataset shape config.
- `requirements-convert.txt`: dependencies needed only for conversion.

## Environment

Use ManiFlow's own conda environment. Do not install ManiFlow into Orbit's `uv` environment because the dependency pins differ.

On RunPod with `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, run the setup script from this repo after cloning it to `/workspace/orbit`:

```bash
cd /workspace/orbit
bash maniflow_orbit_bridge/runpod_setup_maniflow_env.sh
```

If `conda` is missing, the setup script installs Miniconda into `/workspace/miniconda3` first. It then creates a persistent Python 3.10 conda env at `/workspace/conda_envs/maniflow`, clones ManiFlow to `/workspace/maniflow` if needed, installs the minimal Orbit 2D training dependencies, installs ManiFlow editable, and copies the Orbit bridge into ManiFlow. When the same network volume is attached to a new pod, rerunning setup reuses this env instead of recreating it.

RunPod setup overrides:

- `WORKSPACE_DIR`: default `/workspace`.
- `ORBIT_DIR`: default `/workspace/orbit`.
- `MANIFLOW_DIR`: default `/workspace/maniflow`.
- `CONDA_ENV`: default `maniflow`.
- `CONDA_ENV_DIR`: default `/workspace/conda_envs/$CONDA_ENV`; persistent env prefix reused across pod restarts.
- `PYTHON_VERSION`: default `3.10`.
- `MINICONDA_DIR`: default `/workspace/miniconda3`; used only if `conda` is missing.
- `MINICONDA_URL`: default Miniconda Linux x86_64 installer URL.

Manual setup is also possible:

```bash
git clone https://github.com/allenai/maniflow.git /home/vrazer/workspace/orbit/maniflow
cd /home/vrazer/workspace/orbit/maniflow/scripts
conda env create -f conda_environment.yaml
conda activate maniflow
cd ..
pip install -e .
```

For Orbit 2D training, you do not need PyTorch3D, flash-attn, MuJoCo, RoboTwin assets, or DexArt assets unless you plan to use ManiFlow's simulation/pointcloud paths.

If `pip install -r scripts/requirements.txt` hits the known `numpy` conflict, use `numpy==1.24.4` and verify the imports you need:

```bash
python -c "import torch, torchvision, timm, transformers, zarr, numba, maniflow; print('train ok')"
python -c "import numpy, cv2, pandas, pyarrow; print('convert ok')"
```

## Install Bridge Into ManiFlow

Run this from the Orbit repo root:

```bash
python maniflow_orbit_bridge/install_into_maniflow.py \
    --maniflow-dir /home/vrazer/workspace/orbit/maniflow \
    --overwrite
```

This copies files into:

```text
/home/vrazer/workspace/orbit/maniflow/maniflow/dataset/orbit_image_dataset.py
/home/vrazer/workspace/orbit/maniflow/maniflow/config/maniflow_image_orbit.yaml
/home/vrazer/workspace/orbit/maniflow/maniflow/config/robotwin_task/orbit_so100_image.yaml
/home/vrazer/workspace/orbit/maniflow/maniflow/workspace/train_maniflow_orbit_workspace.py
```

You can convert data before or after this copy step. You need this step before training.

## Export Orbit Data To LeRobot

If you already have a LeRobot export like `./dataset/teabags_kitting_50_v2_lerobot`, skip this section.

```bash
uv run bimanual-export-lerobot \
    --input-dir ./dataset/teabags_kitting_50_v2 \
    --output-dir ./dataset/teabags_kitting_50_v2_lerobot \
    --repo-id local/teabags_kitting_50_v2 \
    --fps 30 \
    --video-codec h264 \
    --encoder-threads 4
```

Export notes:

- `--fps 30`: use this if the cameras are 30 FPS. Avoid 60 unless you intentionally want duplicated camera frames.
- `--encoder-threads`: only affects video encoding, not the whole export. Use `4` or `8` first; maxing CPU threads often does not help.
- `--video-codec h264`: safest software codec. Use `h264_nvenc` only if the machine has NVIDIA encoder support and LeRobot/FFmpeg can access it.

## Convert LeRobot To ManiFlow Zarr

The converter can run in the ManiFlow conda env, or any env with `zarr`, `numcodecs`, `pandas`, `pyarrow`, `opencv-python`, and `numpy` installed.

If needed:

```bash
python -m pip install -r maniflow_orbit_bridge/requirements-convert.txt
```

Full conversion:

```bash
python maniflow_orbit_bridge/convert_orbit_lerobot_to_maniflow.py \
    --lerobot-root ./dataset/teabags_kitting_50_v2_lerobot \
    --output-zarr ./dataset/teabags_kitting_50_v2_maniflow.zarr \
    --image-size 224 \
    --overwrite
```

Fast conversion for a smoke dataset:

```bash
python maniflow_orbit_bridge/convert_orbit_lerobot_to_maniflow.py \
    --lerobot-root ./dataset/teabags_kitting_50_v2_lerobot \
    --output-zarr ./dataset/teabags_kitting_50_v2_maniflow_smoke.zarr \
    --image-size 224 \
    --frame-stride 10 \
    --max-episodes 2 \
    --overwrite
```

Converter args:

- `--lerobot-root`: path to the Orbit LeRobot v3 export containing `meta/info.json`, `data/`, and `videos/`.
- `--output-zarr`: output ManiFlow replay buffer directory.
- `--image-size`: square RGB size written to zarr. Keep this at `224` for the default ViT/CLIP config.
- `--frame-stride`: keeps every Nth frame. Use `1` for real training; use larger values only for quick tests.
- `--max-episodes`: converts only the first N episodes for smoke tests.
- `--camera`: optional camera mapping in `NAME=LEROBOT_FEATURE` form. Defaults to `overhead`, `left_wrist`, `right_wrist`.
- `--chunk-length`: zarr chunk length along time. Default `64` is fine.
- `--overwrite`: replace the output zarr if it already exists.

The converter writes:

```text
data/overhead      uint8 [T, 3, H, W]
data/left_wrist    uint8 [T, 3, H, W]
data/right_wrist   uint8 [T, 3, H, W]
data/state         float32 [T, 12]
data/action        float32 [T, 12]
data/task_index    int64 [T]
meta/episode_ends  int64 cumulative episode ends
orbit_tasks.json   task-index to language string mapping
```

Inspect a zarr dataset:

```bash
python - <<'PY'
from pathlib import Path
import json
import zarr

zpath = Path('/home/vrazer/workspace/orbit/dataset/teabags_kitting_50_v2_maniflow.zarr')
root = zarr.open_group(str(zpath), mode='r')
print(root.tree())
ends = root['meta/episode_ends'][:]
print('episodes', len(ends), 'frames', int(ends[-1]))
print(json.loads((zpath / 'orbit_tasks.json').read_text())['task_names'])
PY
```

## Train

Always use the Orbit image-only trainer:

```text
train_maniflow_orbit_workspace.py
```

Do not use ManiFlow's stock `train_maniflow_robotwin_workspace.py` for Orbit, because it imports pointcloud code and may require PyTorch3D even for image-only configs.

On RunPod, use the launcher if your repo is at `/workspace/orbit` and ManiFlow is at `/workspace/maniflow`. If `HF_DATASET_REPO_ID` is set, the launcher downloads the converted `.zarr` dataset from Hugging Face Hub before training:

```bash
cd /workspace/orbit

export HF_TOKEN="<your-hf-token>"
export HF_DATASET_REPO_ID="<your-hf-dataset-repo>"
export HF_DATASET_PATH_IN_REPO="teabags_kitting_50_v2_maniflow.zarr"
export DATASET_NAME="teabags_kitting_50_v2_maniflow.zarr"
export RUN_NAME="maniflow_teabags_v2"
export BATCH_SIZE="16"
export LOGGING_MODE="offline"

bash maniflow_orbit_bridge/runpod_train_maniflow_orbit.sh
```

The launcher writes artifacts to:

```text
/workspace/outputs/train/<RUN_NAME>/
```

Useful RunPod launcher overrides:

- `WORKSPACE_DIR`: default `/workspace`.
- `ORBIT_DIR`: default `/workspace/orbit`.
- `MANIFLOW_DIR`: default `/workspace/maniflow`.
- `DATASET_NAME`: default `teabags_kitting_50_v2_maniflow.zarr` under `/workspace/dataset`.
- `DATASET_ZARR`: full explicit zarr path, overrides `DATASET_NAME`.
- `HF_DATASET_REPO_ID`: optional HF dataset repo ID to download the zarr before training, for example `v-prgmr/teabags-kitting-50-v2-maniflow`.
- `HF_DATASET_REPO_TYPE`: default `dataset`.
- `HF_DATASET_REVISION`: default `main`.
- `HF_DATASET_PATH_IN_REPO`: default `$DATASET_NAME`. Use `.` if the repo root is the zarr contents.
- `HF_DATASET_TOKEN`: optional separate token for dataset download. Defaults to `HF_TOKEN`.
- `HF_DATASET_FORCE_DOWNLOAD`: default `false`; set `true` to pull even when `DATASET_ZARR` already exists.
- `RUN_NAME`: default `maniflow_teabags_v2`.
- `OUTPUT_DIR`: full explicit output path, defaults to `/workspace/outputs/train/$RUN_NAME`.
- `CONDA_ENV`: default `maniflow`.
- `CONDA_ENV_DIR`: default `/workspace/conda_envs/$CONDA_ENV`; training activates this persistent env if present.
- `MINICONDA_DIR`: default `/workspace/miniconda3`; used if `conda` is not already on `PATH`.
- `GPU_DEVICE`: default `cuda:0`.
- `BATCH_SIZE`: default `16`.
- `NUM_WORKERS`: default `4`.
- `NUM_EPOCHS`: default `501`.
- `DEBUG`: default `False`; set `True` for a smoke run.
- `LOGGING_MODE`: default `offline`.
- `PUSH_TO_HF`: default `false`; set `true` to upload artifacts after training exits.
- `PUSH_TO_HF_ON_SUCCESS_ONLY`: default `true`; skip HF upload if training failed.
- `HF_REPO_ID`: required when `PUSH_TO_HF=true`, for example `v-prgmr/maniflow-teabags-v2`.
- `HF_REPO_TYPE`: default `model`.
- `HF_TOKEN`: HF token. If omitted, the current HF CLI login/cache is used.
- `HF_PRIVATE`: default `false`; used when creating the repo if it does not exist.
- `HF_REMOTE_PREFIX`: default `runs/$RUN_NAME` inside the HF repo.
- `HF_UPLOAD_ALL_CHECKPOINTS`: default `false`; uploads only `checkpoints/latest.ckpt` unless true.
- `HF_UPLOAD_WANDB`: default `false`; upload the local W&B folder if true.
- `STOP_POD_ON_EXIT`: default `false`; set `true` to stop the RunPod pod after post-training upload.
- `STOP_POD_ON_SUCCESS_ONLY`: default `true`; only stop if training and artifact upload succeeded.
- `RUNPOD_API_KEY`: required when `STOP_POD_ON_EXIT=true`.
- `RUNPOD_POD_ID`: required when `STOP_POD_ON_EXIT=true`; RunPod may expose this automatically, otherwise set it manually.

RunPod with HF dataset download, artifact upload, and stop-on-success:

```bash
cd /workspace/orbit

export HF_TOKEN="<your-hf-token>"
export HF_DATASET_REPO_ID="<your-hf-dataset-repo>"
export HF_DATASET_PATH_IN_REPO="teabags_kitting_50_v2_maniflow.zarr"
export DATASET_NAME="teabags_kitting_50_v2_maniflow.zarr"
export RUN_NAME="maniflow_teabags_v2"
export BATCH_SIZE="16"
export NUM_WORKERS="4"
export NUM_EPOCHS="501"
export LOGGING_MODE="online"

export PUSH_TO_HF="true"
export HF_REPO_ID="<your-hf-model-repo>"
export HF_REPO_TYPE="model"
export HF_REMOTE_PREFIX="runs/${RUN_NAME}"

export STOP_POD_ON_EXIT="true"
export RUNPOD_API_KEY="<your-runpod-api-key>"
export RUNPOD_POD_ID="<your-pod-id>"

bash maniflow_orbit_bridge/runpod_train_maniflow_orbit.sh
```

The launcher knows training ended because `train_maniflow_orbit_workspace.py` runs as a foreground process. The shell waits for it, captures its exit code, then runs upload/stop logic.

## RunPod Smoke Test

After `runpod_setup_maniflow_env.sh` finishes successfully, run this from the pod to confirm dataset download, bridge install, CUDA/PyTorch imports, and a short ManiFlow debug training loop all work.

For a step-by-step demo that downloads the dataset, trains once, writes `latest.ckpt`, uploads it to HF Hub, then stops the pod, copy from:

```text
/workspace/orbit/runpod_smoke_test_commands.txt
```

Replace the HF values first. If your dataset repo is public, `HF_TOKEN` can be omitted.

```bash
cd /workspace/orbit
git pull

export HF_TOKEN="<your-hf-token>"
export HF_DATASET_REPO_ID="<your-hf-dataset-repo>"
export HF_DATASET_PATH_IN_REPO="teabags_kitting_50_v2_maniflow.zarr"
export DATASET_NAME="teabags_kitting_50_v2_maniflow.zarr"
export RUN_NAME="maniflow_teabags_v2_runpod_smoke"
export DEBUG="True"
export BATCH_SIZE="2"
export NUM_WORKERS="2"
export LOGGING_MODE="offline"
export PUSH_TO_HF="false"
export STOP_POD_ON_EXIT="false"

bash maniflow_orbit_bridge/runpod_train_maniflow_orbit.sh
```

The smoke test passes if the command exits with code `0`, prints `Training exited with code 0`, and creates:

```text
/workspace/outputs/train/maniflow_teabags_v2_runpod_smoke/checkpoints/latest.ckpt
```

The losses only need to be finite for this test. They do not need to improve during a debug smoke run.

Run from the ManiFlow workspace directory:

```bash
conda activate maniflow
cd /home/vrazer/workspace/orbit/maniflow/maniflow/workspace
```

Manual smoke test command:

```bash
python train_maniflow_orbit_workspace.py \
    --config-name=maniflow_image_orbit.yaml \
    robotwin_task=orbit_so100_image \
    robotwin_task.dataset.zarr_path=/home/vrazer/workspace/orbit/dataset/teabags_kitting_50_v2_maniflow.zarr \
    hydra.run.dir=/home/vrazer/workspace/orbit/outputs/train/maniflow_teabags_v2_smoke \
    training.debug=True \
    training.device=cuda:0 \
    dataloader.batch_size=4 \
    val_dataloader.batch_size=4 \
    logging.mode=offline
```

Real training command:

```bash
python train_maniflow_orbit_workspace.py \
    --config-name=maniflow_image_orbit.yaml \
    robotwin_task=orbit_so100_image \
    robotwin_task.dataset.zarr_path=/home/vrazer/workspace/orbit/dataset/teabags_kitting_50_v2_maniflow.zarr \
    hydra.run.dir=/home/vrazer/workspace/orbit/outputs/train/maniflow_teabags_v2 \
    exp_name=maniflow_teabags_v2 \
    training.debug=False \
    training.device=cuda:0 \
    training.num_epochs=501 \
    dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    dataloader.num_workers=4 \
    val_dataloader.num_workers=4 \
    logging.mode=online
```

Training args and common overrides:

- `--config-name=maniflow_image_orbit.yaml`: main installed Hydra config.
- `robotwin_task=orbit_so100_image`: selects the installed Orbit task config.
- `robotwin_task.dataset.zarr_path=...`: required path to the converted Orbit `.zarr`.
- `hydra.run.dir=...`: output directory for `.hydra`, logs, W&B files, and checkpoints.
- `exp_name=...`: readable run name used in W&B grouping.
- `training.debug=True`: smoke mode. ManiFlow forces short training, validation, and frequent checkpoints.
- `training.debug=False`: real training.
- `training.device=cuda:0`: GPU device. Use `cpu` only for tiny sanity checks.
- `training.num_epochs=501`: number of epochs for real runs.
- `dataloader.batch_size`: training batch size. Start with `4`, `8`, or `16` depending on GPU memory.
- `val_dataloader.batch_size`: validation batch size. Usually match train batch size.
- `dataloader.num_workers`: data loader workers. Start with `4`.
- `logging.mode=offline`: local W&B files only.
- `logging.mode=online`: upload to W&B.
- `logging.mode=disabled`: no W&B logging if your installed W&B supports it.

Important typo to avoid:

```text
training.device=cuda:0
```

Do not use:

```text
training.cuda=cuda:0
```

## Config Knobs

Main config:

```text
/home/vrazer/workspace/orbit/maniflow/maniflow/config/maniflow_image_orbit.yaml
```

Task config:

```text
/home/vrazer/workspace/orbit/maniflow/maniflow/config/robotwin_task/orbit_so100_image.yaml
```

Usually adjust these first:

- `dataloader.batch_size`: lower if CUDA OOM.
- `val_dataloader.batch_size`: lower if validation OOM.
- `training.num_epochs`: increase for real training.
- `training.val_every`: validation frequency in epochs.
- `training.checkpoint_every`: checkpoint frequency in epochs.
- `training.sample_every`: how often to log train action MSE.
- `logging.mode`: `offline`, `online`, or `disabled`.

Model size knobs if CUDA OOM persists:

- `policy.n_layer`: default `12`; try `6`.
- `policy.n_emb`: default `768`; try `512`.
- `policy.n_head`: keep `n_emb` divisible by `n_head`.
- `policy.obs_encoder.frozen`: default `true`; keep frozen initially to reduce trainable memory.

Temporal knobs:

- `n_obs_steps`: number of past observation frames. Default `2`.
- `horizon`: action trajectory horizon. Default `16`.
- `n_action_steps`: number of action steps returned/executed. Default `16`.

Dataset knobs in `orbit_so100_image.yaml`:

- `image_shape`: must match converter `--image-size`. Default `[3, 224, 224]`.
- `state` shape: `[12]` for bimanual SO-100.
- `action` shape: `[12]` for bimanual SO-100.
- `cameras`: defaults to `overhead`, `left_wrist`, `right_wrist`.
- `load_to_memory`: default `false`; keep false for large image zarrs.

## Outputs

A successful run creates:

```text
outputs/train/<run_name>/
├── .hydra/config.yaml
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=....ckpt
├── train_maniflow_orbit_workspace.log
└── wandb/
```

Smoke-test expectations:

- Checkpoints should appear quickly.
- Losses should be finite, not `nan` or `inf`.
- Loss does not need to improve in a smoke test. A debug run may only run tens of optimizer steps.

Real-training expectations:

- `val_loss` should trend down over many epochs.
- `train_action_mse_error` should trend down over many epochs.
- Flat/noisy loss over 30 steps is not meaningful; flat/noisy loss over thousands of steps is a problem.

## Data Semantics

The policy predicts Orbit's absolute commanded joint targets in the exported action order:

```text
left shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper,
right shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The ManiFlow dataset supplies observations as:

```text
overhead + left_wrist + right_wrist + agent_pos + task_name
```

The default config uses `ManiFlowTransformerImagePolicy` with language conditioning enabled and a multi-camera ViT/CLIP visual encoder.

## Troubleshooting

If `cv2` fails with a NumPy ABI error:

```bash
python -m pip uninstall -y numpy opencv-python opencv-python-headless
python -m pip install "numpy==1.24.4" "opencv-python==4.5.5.64"
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

If parquet reading fails with `KeyError: 'observation'`, install `pyarrow` and remove `fastparquet` if needed:

```bash
python -m pip install "pyarrow==15.0.2"
python -m pip uninstall -y fastparquet
```

If training asks for PyTorch3D, you are using the wrong train script. Use:

```text
train_maniflow_orbit_workspace.py
```

If `import torch` fails with `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`, rerun the RunPod setup script. It pins `mkl<2024.1` and `intel-openmp<2024.1` to avoid the PyTorch/MKL runtime mismatch:

```bash
cd /workspace/orbit
bash maniflow_orbit_bridge/runpod_setup_maniflow_env.sh
```

If setup fails with `ModuleNotFoundError: No module named 'maniflow'`, pull the latest Orbit repo and rerun setup. The setup script creates ManiFlow's missing `maniflow/__init__.py` before editable install:

```bash
cd /workspace/orbit
git pull
bash maniflow_orbit_bridge/runpod_setup_maniflow_env.sh
```

If Hydra cannot find the config, reinstall the bridge:

```bash
python /home/vrazer/workspace/orbit/maniflow_orbit_bridge/install_into_maniflow.py \
    --maniflow-dir /home/vrazer/workspace/orbit/maniflow \
    --overwrite
```

If CUDA runs out of memory, lower batch size first:

```bash
dataloader.batch_size=4 val_dataloader.batch_size=4
```

If OOM persists, reduce model size:

```bash
policy.n_layer=6 policy.n_emb=512 policy.n_head=8
```
