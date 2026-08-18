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

## Setup

```bash
bash lewm_orbit_bridge/setup_lewm.sh
```

Override locations with `LEWM_ROOT`, `STABLEWM_ROOT`, or `LEWM_VENV`. The scripts default to `external/le-wm`, `external/stable-worldmodel`, and `.venv-lewm`.

## Convert and Split

Collection type can be explicit with `TYPE=/path`. Explicit types are recommended so no path-name inference is involved.

```bash
.venv-lewm/bin/python -m lewm_orbit_bridge.convert_orbit_to_lewm \
  --dataset expert=/workspace/orbit/dataset/teabags_kitting_50_v2 \
  --dataset policy_success=/workspace/orbit/dataset/maniflow_rollouts/successes \
  --dataset policy_failure=/workspace/orbit/dataset/maniflow_rollouts/failures \
  --dataset hil/rac_correction=/workspace/orbit/dataset/maniflow_hil_bounded \
  --output /workspace/lewm_data/teabag.lance \
  --seed 3072 \
  --overwrite
```

The converter creates:

```text
teabag.lance/
teabag.splits.json
teabag.manifest.json
```

The 80/10/10 train/validation/test split is episode-level, reproducible, and stratified by collection type where category size permits. Exact stable episode UIDs are persisted. Test episodes are not used for training, model selection, normalization, or successful-goal construction.

## Validate Alignment

```bash
.venv-lewm/bin/python -m lewm_orbit_bridge.validate_lewm_dataset \
  /workspace/lewm_data/teabag.lance \
  --report /workspace/lewm_data/teabag.validation.json \
  --visual-sample /workspace/lewm_data/teabag.transitions.png
```

Inspect `teabag.transitions.png` before training. It shows `observation_t`, the exact accepted `action_t`, and `observation_t+1`.

## Configure Paths

The defaults in `config/teabag.yaml` can be overridden without editing YAML:

```bash
export LEWM_DATASET=/workspace/lewm_data/teabag.lance
export LEWM_SPLITS=/workspace/lewm_data/teabag.splits.json
export LEWM_OUTPUT_DIR=/workspace/outputs/lewm/teabag_overhead_fs3_v1
export LEWM_ROOT=/workspace/orbit/external/le-wm
```

## Smoke and Tiny Overfit

```bash
bash lewm_orbit_bridge/train_teabag_lewm.sh --mode smoke --max-steps 50
bash lewm_orbit_bridge/train_teabag_lewm.sh --mode tiny-overfit --max-steps 400
```

Smoke mode checks finite forward/backward losses, optimizer stepping, checkpoint export, and peak GPU allocation. Tiny-overfit mode uses two training episodes and fails unless final prediction loss is at least 20% below initial prediction loss.

Training order is deliberately leak-free:

```text
split episodes
    -> fit z-score action normalization on TRAIN episodes only
    -> construct train/validation clip subsets
    -> train
```

## Full Training

```bash
bash lewm_orbit_bridge/train_teabag_lewm.sh --mode full
```

Defaults are 100 epochs, batch 32, BF16 mixed precision, AdamW, learning rate `5e-5`, weight decay `1e-3`, image size 224, embedding dimension 192, history 3, one-step prediction, and SIGReg weight `0.09`. The ViT is initialized from scratch.

Outputs include the resolved config, train-only action statistics, exact splits, Orbit/LeWM SHAs, Lightning checkpoints, object/weights exports, CSV curves, and mode summaries.

## Held-Out Evaluation

```bash
export LEWM_CHECKPOINT="$LEWM_OUTPUT_DIR/lewm_object.ckpt"
bash lewm_orbit_bridge/eval_teabag_lewm.sh
```

Latent prediction reports 1/3/5/10-transition MSE and cosine similarity on test episodes. It compares:

1. LeWM with the correct executed action sequence
2. LeWM with a whole action sequence shuffled from another held-out window
3. Persistence, `z_hat(t+h) = z_t`

The primary action-conditioning criterion is positive shuffled-action margin:

```text
MSE(shuffled actions) - MSE(correct actions) > 0
```

If correct and shuffled actions perform similarly, LeWM is not sufficiently action-sensitive for future ManiFlow candidate reranking even if it beats persistence.

Goal-structure evaluation constructs nearest and mean successful-goal embeddings from expert/genuinely successful **training** episodes, then measures test trajectories. These distances are exploratory and are not treated as calibrated values.

## Go/No-Go Order

1. Training is stable and tiny-subset overfitting succeeds.
2. Test latent prediction beats persistence.
3. Correct actions beat shuffled actions.
4. Autoregressive degradation over 1/3/5/10 transitions is graceful.
5. Successful-goal latent distance behaves sensibly.

Good-versus-bad continuation pairing, CEM, MPC, ManiFlow reranking, TOPReward integration, multi-camera input, and online control are intentionally out of scope for this branch.
