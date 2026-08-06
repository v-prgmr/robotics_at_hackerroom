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
  --episode episode-000001 \
  --batch-size 1
```

Then resume and score everything else:

```bash
bash scripts/run_topreward_runpod.sh --batch-size 1
```

`--batch-size 1` is the lowest-VRAM mode and scores the two answer candidates
sequentially. Use `--batch-size 2` only if the GPU has enough memory to score a
True/False or Yes/No pair together. Prefix anchors and episodes are always
processed sequentially.

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
