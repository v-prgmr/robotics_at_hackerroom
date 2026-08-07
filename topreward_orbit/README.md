# Orbit TOPReward on RunPod

This scorer reads Orbit intermediate recordings directly and never modifies the
source dataset. It uses 15 trajectory-prefix anchors by default. Every prefix is
sampled independently to at most 16 temporally spaced frames before Qwen3-VL
inference.

Expected data location:

```text
/workspace/orbit/dataset/topReward_smoketest/
├── teabags_kitting_50_v2/
├── maniflow_hil_bounded/
├── maniflow_rollout_successes/
└── maniflow_rollout_failures/
```

Set up the isolated environment:

```bash
cd /workspace/orbit
bash scripts/setup_topreward_runpod.sh
```

Score one episode first:

```bash
bash scripts/run_topreward_runpod.sh \
  --dataset teabags_kitting_50_v2 \
  --episode episode-000001
```

Then resume and score everything else:

```bash
export STOP_POD_ON_EXIT="true"
export STOP_POD_ON_SUCCESS_ONLY="true"
export RUNPOD_API_KEY="<your-runpod-api-key>"
export RUNPOD_POD_ID="${POD_ID:-<your-pod-id>}"

bash scripts/run_topreward_runpod.sh
```

Automatic shutdown is opt-in. By default, the pod stops only after scoring and
aggregate output generation succeed. Set `STOP_POD_ON_SUCCESS_ONLY=false` if it
should also stop after a failed or interrupted run. `RUNPOD_POD_ID` falls back
to RunPod's `POD_ID` environment variable when available.

Each prefix requires one model forward. True/False probabilities are read from
the same next-token distribution; the video is not duplicated. Prefix anchors
and episodes are processed sequentially. The scorer rejects resume attempts
whose model, FPS, prompts, or sampling settings do not match `config.json`.
An episode-level progress bar reports completed episodes, throughput, and ETA;
resumed episodes count toward the displayed total.

Results are written under `/workspace/outputs/topreward_smoketest` by default:

- `episode_scores.parquet`: terminal raw scores, prefix margin change, grouped metadata, and policy-only success score.
- `anchor_scores.parquet`: raw True/False scores and within-episode plot normalization at each prefix anchor.
- `timestep_scores.parquet`: raw anchor scores linearly interpolated onto every action timestep.
- `summary.json`: VOC and terminal raw-score summaries separated by collection type.
- `episodes/`: resumable per-episode JSON results.

Min-max-normalized values exist only in `anchor_scores.parquet` for within-episode
plots. Cross-episode comparisons and future ManiFlow weighting must use raw
log-probabilities or margins.

The success score is computed only across labeled policy successes/failures as:

```text
z(prefix_margin_change) + z(full_video_yes_no_margin)
```

HIL episodes remain `hil_correction`; their recording `success=true` value is not
treated as full-task success.

## Browse scores in Rerun

After copying the Orbit dataset and TOPReward output to the same machine, browse
one dataset root with the existing lazy companion window:

```bash
uv run bimanual-dataset-rerun \
  --dataset dataset/topReward_smoketest/teabags_kitting_50_v2 \
  --topreward-dir outputs/topreward_smoketest \
  --lazy
```

The viewer shows camera frames beside raw/interpolated TOPReward curves and
episode-local normalized curves. Measured anchors are logged separately from
the interpolated values. Use the companion window's Previous, Reload, and Next
buttons or Left/Right, `r`, and `q` shortcuts.
