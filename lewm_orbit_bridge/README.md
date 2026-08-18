# Orbit LeWorldModel Experiment

This bridge evaluates whether stock [LeWorldModel](https://github.com/lucas-maes/le-wm) learns useful action-conditioned latent dynamics from Orbit teabag-kitting trajectories. It is isolated from ManiFlow and TOPReward and does not implement planning, candidate reranking, control, or policy updates.

## Pinned Sources

- Orbit base: `253f367e9aa32faf0cca5935caf0de2bc6b0e8c2`, before the RISE progress-head commits
- LeWM: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- Stable World Model: `9a66d7d020043c8efb507f45373e808714f0842d`

`setup_lewm.sh` clones both external repositories into ignored directories and checks out these exact commits without modifying either checkout.

## Data Semantics

The converter preserves Orbit's established intermediate/LeRobot row semantics:

```text
recorded observation[t]
    -> same-row accepted action[t], after clipping/safety checks
    -> recorded observation[t+1]
```

All collections, including expert teleoperation, preserve same-row observation/action pairing and every source row. Camera capture is asynchronous: calling `match()` after command dispatch does not imply that the selected buffered frame was exposed afterward. The conversion manifest records frame-age and command-duration diagnostics, but does not infer exact exposure ordering from call order.

The stored action is left arm followed by right arm, with all dimensions and both grippers retained. The converter rejects action-name or dimensionality changes rather than truncating or reordering values. It also requires the pinned external checkouts so manifests cannot claim an unobserved upstream revision.

Stable World Model applies `frameskip=3` while loading and groups all three raw actions into one model action. A raw 12-D action therefore becomes 36-D at each model transition. Repeated camera references are valid because the control loop can reuse a 30 Hz camera frame; camera and video indices must be monotonically non-decreasing, not strictly increasing.

HIL/RaC `success=true` means the correction clip was saved. It is never interpreted as full-task success.

## Prerequisites

- Linux with `git` and [`uv`](https://docs.astral.sh/uv/) available on `PATH`
- Python 3.10, installed automatically by `uv` when needed
- An NVIDIA GPU with BF16 support for the configured training run
- Orbit intermediate-format source datasets with `episode-*/timesteps.parquet`, `episode_metadata.json`, and `videos/overhead.mp4`
- Enough storage for the source videos, the converted Lance dataset, checkpoints, and plots

Run every command below from the Orbit repository root. The launchers compute `ORBIT_DIR` automatically, but setting it explicitly makes copied commands unambiguous:

```bash
cd /workspace/orbit
export ORBIT_DIR="$(pwd)"
```

## Environment Variables

The bridge supports the following custom environment variables. Paths may be absolute or relative to the current working directory, but absolute paths are recommended for remote training machines.

| Variable | Default | Used by | Purpose |
| --- | --- | --- | --- |
| `ORBIT_DIR` | Repository root inferred from the launcher | Setup, train, eval | Orbit checkout containing `lewm_orbit_bridge/` |
| `LEWM_ROOT` | `$ORBIT_DIR/external/le-wm` | Setup, config, train, eval | Unmodified LeWM checkout |
| `LEWM_SHA` | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | Setup | LeWM commit checked out in detached mode |
| `STABLEWM_ROOT` | `$ORBIT_DIR/external/stable-worldmodel` | Setup and conversion CLI | Stable World Model checkout providing Lance support |
| `STABLEWM_SHA` | `9a66d7d020043c8efb507f45373e808714f0842d` | Setup | Stable World Model commit checked out in detached mode |
| `LEWM_VENV` | `$ORBIT_DIR/.venv-lewm` | Setup | Python 3.10 virtual environment created by `uv` |
| `LEWM_PYTHON` | `$ORBIT_DIR/.venv-lewm/bin/python` | Train and eval launchers | Python executable used after setup |
| `LEWM_CONFIG` | `$ORBIT_DIR/lewm_orbit_bridge/config/teabag.yaml` | Train and eval launchers | LeWM experiment configuration |
| `LEWM_DATASET` | No default; required | Hydra config | Converted Lance dataset path |
| `LEWM_SPLITS` | No default; required | Hydra config | Exact episode-level split manifest |
| `LEWM_OUTPUT_DIR` | `outputs/lewm/teabag_overhead_fs3_v1` | Hydra config | Current run's checkpoints, metadata, curves, and evaluation outputs |
| `LEWM_CHECKPOINT` | No default; required for eval | Eval launcher | Trained `lewm_object.ckpt` selected by validation loss |
| `LEWM_RESUME_CHECKPOINT` | No default | Training | Full-state Lightning checkpoint used to restore model, optimizer, scheduler, epoch, and step |
| `STABLEWM_HOME` | `~/.stable-wm` | Stable World Model | Optional upstream cache root; it does not replace required `LEWM_DATASET` and `LEWM_SPLITS` values |
| `HF_UPLOAD_ENABLED` | `false` | Training | Enable periodic recovery and final run uploads to Hugging Face Hub |
| `HF_REPO_ID` | No default; required when enabled | Training | Target Hugging Face model repository, for example `org/teabag-lewm` |
| `HF_TOKEN` | Falls back to `HUGGING_FACE_HUB_TOKEN` | Training | Hugging Face write token; never persisted in run artifacts |
| `HF_PRIVATE` | `true` | Training | Create the target model repository as private when it does not exist |
| `HF_CHECKPOINT_INTERVAL_EPOCHS` | `5` | Training | Periodic full-state upload cadence; `0` means final upload only |
| `HF_REMOTE_PREFIX` | `runs/$experiment_name` | Training | Path inside the model repository for this run |
| `HF_UPLOAD_RETRIES` | `3` | Training | Attempts for each Hub upload operation |
| `HF_UPLOAD_RETRY_DELAY_S` | `10` | Training | Delay between failed Hub upload attempts |

If `LEWM_VENV` is customized, also set `LEWM_PYTHON` because the train and evaluation launchers do not derive one variable from the other:

```bash
export LEWM_VENV=/workspace/venvs/lewm
export LEWM_PYTHON="$LEWM_VENV/bin/python"
```

`LEWM_SHA` and `STABLEWM_SHA` are exposed for setup reproducibility, but the converter intentionally requires the pinned revisions listed above. Updating either revision requires updating and revalidating the bridge's corresponding pin check.

Optional Weights & Biases logging uses standard W&B variables such as `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_MODE`. It remains disabled unless training is launched with `wandb.enabled=true`.

## Step 1: Configure Paths

Choose persistent locations for the converted data and experiment outputs. This example keeps external repositories and the Python environment under Orbit while placing large artifacts under `/workspace`:

```bash
export LEWM_ROOT="$ORBIT_DIR/external/le-wm"
export STABLEWM_ROOT="$ORBIT_DIR/external/stable-worldmodel"
export LEWM_VENV="$ORBIT_DIR/.venv-lewm"
export LEWM_PYTHON="$LEWM_VENV/bin/python"
export LEWM_CONFIG="$ORBIT_DIR/lewm_orbit_bridge/config/teabag.yaml"

export LEWM_DATASET=/workspace/lewm_data/teabag.lance
export LEWM_SPLITS=/workspace/lewm_data/teabag.splits.json
```

Create only the parent directory. The converter creates the Lance table itself:

```bash
mkdir -p "$(dirname "$LEWM_DATASET")"
```

## Step 2: Install the Environment

The setup script clones both pinned upstream repositories, checks out exact detached commits, creates the Python 3.10 environment, installs Stable World Model's training dependencies and the conversion dependencies, and performs import checks. Simulation environment extras are intentionally omitted because this experiment does not use online Gymnasium environments:

```bash
bash lewm_orbit_bridge/setup_lewm.sh
```

Verify the environment, revisions, and GPU visibility:

```bash
"$LEWM_PYTHON" -c 'import lancedb, stable_pretraining, stable_worldmodel, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)'
git -C "$LEWM_ROOT" rev-parse HEAD
git -C "$STABLEWM_ROOT" rev-parse HEAD
```

Expected revisions are listed under [Pinned Sources](#pinned-sources). Do not continue if imports fail or the revisions differ.

Warnings that optional environments such as ALE are unavailable are expected and do not affect Lance conversion, LeWM training, or latent evaluation.

## Step 3: Verify Source Datasets

Set or inspect the four source roots before conversion:

```bash
test -d "$ORBIT_DIR/dataset/teabags_kitting_50_v2"
test -d "$ORBIT_DIR/dataset/maniflow_rollouts/successes"
test -d "$ORBIT_DIR/dataset/maniflow_rollouts/failures"
test -d "$ORBIT_DIR/dataset/maniflow_hil_bounded"
```

Use explicit `COLLECTION_TYPE=/path` arguments. This avoids relying on directory-name inference and preserves the intended provenance labels.

## Step 4: Convert and Split

Run the converter with the dedicated Python environment:

```bash
"$LEWM_PYTHON" -m lewm_orbit_bridge.convert_orbit_to_lewm \
  --dataset "expert=$ORBIT_DIR/dataset/teabags_kitting_50_v2" \
  --dataset "policy_success=$ORBIT_DIR/dataset/maniflow_rollouts/successes" \
  --dataset "policy_failure=$ORBIT_DIR/dataset/maniflow_rollouts/failures" \
  --dataset "hil/rac_correction=$ORBIT_DIR/dataset/maniflow_hil_bounded" \
  --output "$LEWM_DATASET" \
  --splits-output "$LEWM_SPLITS" \
  --lewm-root "$LEWM_ROOT" \
  --stablewm-root "$STABLEWM_ROOT" \
  --seed 3072 \
  --overwrite
```

The converter writes one episode-contiguous Lance dataset plus reproducibility sidecars:

```text
/workspace/lewm_data/teabag.lance/
/workspace/lewm_data/teabag.splits.json
/workspace/lewm_data/teabag.manifest.json
```

The split is 80% train, 10% validation, and 10% test at episode level, stratified by collection type where counts permit. The test episode IDs are never used for training, normalization, model selection, or successful-goal construction. `--overwrite` replaces a previous conversion; omit it when accidental replacement should fail.

Inspect the generated manifest before continuing:

```bash
"$LEWM_PYTHON" -c 'import json, os; p=os.environ["LEWM_DATASET"].removesuffix(".lance") + ".manifest.json"; d=json.load(open(p)); print(json.dumps({k:d[k] for k in ("num_episodes", "num_timesteps", "action_dim", "collection_type_counts", "split_counts", "frameskip")}, indent=2))'
```

Confirm that `action_dim` is 12 and review the collection and split counts.

## Step 5: Validate Action/Frame Alignment

Run strict validation and generate a transition contact sheet:

```bash
"$LEWM_PYTHON" -m lewm_orbit_bridge.validate_lewm_dataset \
  "$LEWM_DATASET" \
  --report /workspace/lewm_data/teabag.validation.json \
  --visual-sample /workspace/lewm_data/teabag.transitions.png \
  --sample-count 8 \
  --seed 3072
```

Validation must exit successfully. Manually inspect `/workspace/lewm_data/teabag.transitions.png`; each row shows `observation_t`, the exact accepted `action_t`, and `observation_t+1`. Also inspect the conversion manifest's `camera_timing_diagnostic`. It is diagnostic metadata, not a reason to shift actions between recorder rows.

Do not start GPU training until validation passes and the contact sheet looks plausible.

## Step 6: Run the Training Smoke Test

Use a separate output directory so smoke artifacts cannot be mistaken for the full run:

```bash
export LEWM_OUTPUT_DIR=/workspace/outputs/lewm/teabag_overhead_fs3_smoke

bash lewm_orbit_bridge/train_teabag_lewm.sh \
  --mode smoke \
  --max-steps 50 \
  --batch-size 32 \
  --num-workers 6
```

Inspect the smoke summary:

```bash
"$LEWM_PYTHON" -m json.tool "$LEWM_OUTPUT_DIR/smoke_summary.json"
test -f "$LEWM_OUTPUT_DIR/lewm_object.ckpt"
```

The run must complete forward and backward passes, report finite initial/final prediction losses, take optimizer steps, save a checkpoint, and report stable peak GPU allocation. If GPU memory is insufficient, retry with `--batch-size 16` or `--batch-size 8`.

## Step 7: Overfit Two Episodes

Use another isolated output directory:

```bash
export LEWM_OUTPUT_DIR=/workspace/outputs/lewm/teabag_overhead_fs3_tiny_overfit

bash lewm_orbit_bridge/train_teabag_lewm.sh \
  --mode tiny-overfit \
  --max-steps 400 \
  --batch-size 32 \
  --num-workers 6
```

Inspect the result:

```bash
"$LEWM_PYTHON" -m json.tool "$LEWM_OUTPUT_DIR/tiny-overfit_summary.json"
```

The command fails intentionally if final prediction loss is not at least 20% below initial prediction loss. Do not launch the 100-epoch run until this test succeeds.

Training is leak-free in every mode:

```text
split episodes
    -> fit z-score action normalization on TRAIN episodes only
    -> construct train/validation clip subsets
    -> train
```

## Step 8: Launch the Full Experiment

Select the final output directory. To enable private Hugging Face uploads, configure the repository, token, and desired periodic checkpoint interval before launching:

```bash
export LEWM_OUTPUT_DIR=/workspace/outputs/lewm/teabag_overhead_fs3_v1
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID=your-org/teabag-lewm
export HF_TOKEN=hf_your_write_token
export HF_PRIVATE=true
export HF_CHECKPOINT_INTERVAL_EPOCHS=5
export HF_REMOTE_PREFIX=runs/teabag_overhead_fs3_v1

bash lewm_orbit_bridge/train_teabag_lewm.sh \
  --mode full \
  --batch-size 96 \
  --num-workers 6
```

Defaults come from `lewm_orbit_bridge/config/teabag.yaml`: BF16 mixed precision, AdamW, learning rate `5e-5`, weight decay `1e-3`, image size 224, embedding dimension 192, history size 3, one predicted transition, frameskip 3, and SIGReg weight `0.09`. The ViT starts from random weights.

Known command-line options are `--mode`, `--max-steps`, `--batch-size`, `--num-workers`, and `--resume-checkpoint`. Additional arguments are interpreted as OmegaConf dot-list overrides. Examples:

```bash
bash lewm_orbit_bridge/train_teabag_lewm.sh --mode full loader.batch_size=64
bash lewm_orbit_bridge/train_teabag_lewm.sh --mode full trainer.max_epochs=10 wandb.enabled=true wandb.project=orbit-lewm
```

Prefer `--batch-size` over `loader.batch_size=...` when only changing batch size.

Keep full training epoch-based (`trainer.max_epochs=100`, `trainer.max_steps=-1`) so every epoch reaches validation and checkpoint selection. For the current 70-episode training split with 47,203 valid clips, batch size 96 on one GPU gives 491 full batches per epoch and approximately 49,100 optimizer steps over 100 epochs. This is the batch-96 equivalent of the released 100-epoch LeWM schedule; use `--max-steps` only for smoke and tiny-overfit diagnostics.

Full mode displays one TQDM bar across all estimated optimizer steps, including percentage, elapsed time, remaining ETA, throughput, and current epoch. The total is calculated from the configured data loader and trainer rather than hard-coded; resumed runs initialize the bar from the restored global step. Smoke and tiny-overfit modes retain Lightning's normal per-epoch progress bar.

Local `checkpoints/last.ckpt` and validation-selected `checkpoints/best.ckpt` are updated every epoch. When HF upload is enabled, `HF_CHECKPOINT_INTERVAL_EPOCHS=N` additionally uploads resumable recovery files every `N` epochs:

```text
$HF_REMOTE_PREFIX/recovery/last.ckpt
$HF_REMOTE_PREFIX/recovery/best.ckpt
```

`last.ckpt` contains optimizer and scheduler state and is the checkpoint to use for resuming. Setting the interval to `0` disables periodic network uploads but still uploads the complete run after successful training. Periodic upload failures are retried and then reported as warnings so a temporary network outage does not discard the training run; a failed final upload exits with an error while leaving all local artifacts intact.

The selected output directory contains:

```text
action_normalization.json       train-only action statistics
checkpoints/                    Lightning best and last checkpoints
dataset_statistics.json         copied conversion manifest
git_versions.json               Orbit and LeWM revisions
lewm_object.ckpt                validation-selected model for evaluation
lewm_weights.pt                 validation-selected state dict
resolved_config.yaml            exact resolved training configuration
run_manifest.json               artifact hashes and selected checkpoint
splits.json                     exact train/validation/test episode IDs
training_curves/                CSV metrics
full_summary.json               run summary and peak GPU memory
```

After successful training, the entire output directory is uploaded under `HF_REMOTE_PREFIX`, including the validation-selected object and weights checkpoints, available Lightning recovery checkpoints, resolved configuration, split manifest, train-only normalization, dataset statistics, hashes, summaries, and CSV curves. Use a unique prefix for each independent run so files from an older run cannot remain under the same Hub path.

## Resume After Instance Loss

Download the remote recovery and historical best checkpoints on the replacement instance. Use the same absolute `LEWM_OUTPUT_DIR`, dataset, split manifest, and configuration as the original run:

```bash
mkdir -p "$LEWM_OUTPUT_DIR/recovery" "$LEWM_OUTPUT_DIR/checkpoints"

hf download "$HF_REPO_ID" \
  "$HF_REMOTE_PREFIX/recovery/last.ckpt" \
  --repo-type model \
  --token "$HF_TOKEN" \
  --local-dir /workspace/hf-recovery

export LEWM_RESUME_CHECKPOINT="/workspace/hf-recovery/$HF_REMOTE_PREFIX/recovery/last.ckpt"

hf download "$HF_REPO_ID" \
  "$HF_REMOTE_PREFIX/recovery/best.ckpt" \
  --repo-type model \
  --token "$HF_TOKEN" \
  --local-dir /workspace/hf-recovery

cp "/workspace/hf-recovery/$HF_REMOTE_PREFIX/recovery/best.ckpt" \
  "$LEWM_OUTPUT_DIR/checkpoints/best.ckpt"

bash lewm_orbit_bridge/train_teabag_lewm.sh \
  --mode full \
  --batch-size 32 \
  --num-workers 6
```

You can use `--resume-checkpoint /path/to/last.ckpt` instead of `LEWM_RESUME_CHECKPOINT`. The trainer verifies hashes for the resolved config, split manifest, action normalization, and dataset manifest before accepting a checkpoint. Do not resume from `lewm_object.ckpt` or `lewm_weights.pt`; those files do not contain optimizer, scheduler, epoch, or global-step state.

## Step 9: Evaluate the Held-Out Test Set

Evaluation verifies hashes in `run_manifest.json`, so `LEWM_OUTPUT_DIR` and `LEWM_CHECKPOINT` must refer to the same full run:

```bash
export LEWM_OUTPUT_DIR=/workspace/outputs/lewm/teabag_overhead_fs3_v1
export LEWM_CHECKPOINT="$LEWM_OUTPUT_DIR/lewm_object.ckpt"

bash lewm_orbit_bridge/eval_teabag_lewm.sh
```

The wrapper runs both held-out evaluations with default settings. It intentionally accepts no extra arguments. For custom latent horizons or window counts, invoke the evaluator directly:

```bash
"$LEWM_PYTHON" -m lewm_orbit_bridge.evaluate_latent_prediction \
  --config "$LEWM_CONFIG" \
  --checkpoint "$LEWM_CHECKPOINT" \
  --horizons 1 3 5 10 \
  --max-windows 1000 \
  --seed 3072
```

For a different goal-trajectory sampling stride:

```bash
"$LEWM_PYTHON" -m lewm_orbit_bridge.evaluate_goal_structure \
  --config "$LEWM_CONFIG" \
  --checkpoint "$LEWM_CHECKPOINT" \
  --sample-stride 30
```

Latent prediction reports MSE and cosine similarity at 1, 3, 5, and 10 world-model transitions. It compares:

1. LeWM using the correct executed action sequence
2. LeWM using a whole action sequence from a different held-out episode
3. Persistence, `z_hat(t+h) = z_t`

The primary action-conditioning criterion is:

```text
MSE(shuffled actions) - MSE(correct actions) > 0
```

If correct and shuffled actions perform similarly, LeWM is not sufficiently action-sensitive for future ManiFlow candidate reranking even if it beats persistence. Goal-structure evaluation constructs nearest and mean successful-goal embeddings from expert/genuinely successful training episodes and measures test trajectories; these distances are exploratory, not calibrated values.

Evaluation outputs are written under:

```text
$LEWM_OUTPUT_DIR/latent_prediction/
$LEWM_OUTPUT_DIR/goal_structure/
```

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `LEWM_CHECKPOINT` is required | Export it to the full run's `lewm_object.ckpt` before evaluation |
| Checkpoint or artifact hash mismatch | Ensure `LEWM_OUTPUT_DIR`, `LEWM_CONFIG`, and `LEWM_CHECKPOINT` all refer to the same run |
| CUDA unavailable or BF16 unsupported | Use a CUDA-capable training host; experiment 1 is configured for GPU BF16 |
| GPU out of memory | Lower `--batch-size`; do not change image size or architecture for the first experiment |
| DataLoader worker failure | Retry with `--num-workers 0` to diagnose, then increase gradually |
| Existing Lance output error | Add `--overwrite` only when intentionally regenerating all data and splits |
| Unexpected upstream commit | Rerun `setup_lewm.sh`; the converter rejects unpinned LeWM or Stable World Model revisions |
| Parquet nested-list read errors under system Python | Use `$LEWM_PYTHON`, not an unrelated Python installation |
| Tiny-overfit loss does not fall by 20% | Stop and debug alignment, normalization, or data loading before full training |
| Periodic HF upload warning | Training continues after retries; verify network/token and let the final upload retry, or upload the local output directory manually |
| Resume checkpoint missing | Download `$HF_REMOTE_PREFIX/recovery/last.ckpt` and set `LEWM_RESUME_CHECKPOINT` to the resulting local file |

## Go/No-Go Order

1. Training is stable and tiny-subset overfitting succeeds.
2. Test latent prediction beats persistence.
3. Correct actions beat shuffled actions.
4. Autoregressive degradation over 1/3/5/10 transitions is graceful.
5. Successful-goal latent distance behaves sensibly.

Good-versus-bad continuation pairing, CEM, MPC, ManiFlow reranking, TOPReward integration, multi-camera input, and online control are intentionally out of scope for this branch.
