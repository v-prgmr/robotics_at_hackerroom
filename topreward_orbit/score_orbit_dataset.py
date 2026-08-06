"""Score Orbit intermediate datasets with TOPReward on Qwen3-VL.

The input is an Orbit dataset container whose children are dataset roots. Each
dataset root contains ``episode-*`` directories with metadata, timesteps, and
an overhead video. Results are written as sidecars; source recordings are never
modified.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SCORER_VERSION = 3
DEFAULT_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the following task: "
)
DEFAULT_BOOLEAN_SUFFIX = (
    "{instruction} Decide whether the above statement is True or not. The answer is: {answer}"
)
DEFAULT_YES_NO_SUFFIX = (
    "Did the robot manipulation trajectory in the above video successfully complete the following task: "
    "{instruction} Answer Yes or No. The answer is: {answer}"
)


@dataclass(frozen=True)
class PrefixSpec:
    anchor_index: int
    anchor_timestep_index: int
    anchor_video_frame_index: int
    sampled_timestep_indices: list[int]
    sampled_video_frame_indices: list[int]


def uniformly_spaced_indices(start: int, stop: int, count: int) -> list[int]:
    """Return up to ``count`` unique indices spanning the inclusive interval."""

    if count < 1:
        raise ValueError("count must be >= 1")
    if stop < start:
        raise ValueError(f"stop must be >= start, got {start=} {stop=}")
    length = stop - start + 1
    if length <= count:
        return list(range(start, stop + 1))
    return np.unique(np.linspace(start, stop, count).round().astype(np.int64)).tolist()


def build_prefix_specs(
    timesteps: pd.DataFrame,
    *,
    camera: str,
    num_anchors: int,
    max_frames: int,
) -> list[PrefixSpec]:
    """Build prefix anchors and a separate capped temporal sample per prefix."""

    if num_anchors < 1:
        raise ValueError("num_anchors must be >= 1")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    frame_column = f"{camera}_video_frame_index"
    required = {"timestep_index", frame_column}
    missing = required - set(timesteps.columns)
    if missing:
        raise ValueError(f"Missing timestep columns: {sorted(missing)}")
    if timesteps.empty:
        raise ValueError("Episode has no timesteps")

    ordered = timesteps.sort_values("timestep_index").reset_index(drop=True)
    scoring_frame_indices = ordered[frame_column].ffill().bfill()
    if bool(scoring_frame_indices.isna().all()):
        raise ValueError(f"Episode contains no valid {frame_column} values")
    anchors = uniformly_spaced_indices(0, len(ordered) - 1, num_anchors)
    specs: list[PrefixSpec] = []
    for anchor_index, anchor_position in enumerate(anchors):
        sampled_positions = uniformly_spaced_indices(0, anchor_position, max_frames)
        sampled_rows = ordered.iloc[sampled_positions]
        frame_indices = scoring_frame_indices.iloc[sampled_positions]
        anchor_row = ordered.iloc[anchor_position]
        specs.append(
            PrefixSpec(
                anchor_index=anchor_index,
                anchor_timestep_index=int(anchor_row["timestep_index"]),
                anchor_video_frame_index=int(scoring_frame_indices.iloc[anchor_position]),
                sampled_timestep_indices=[int(value) for value in sampled_rows["timestep_index"]],
                sampled_video_frame_indices=[int(value) for value in frame_indices],
            )
        )
    return specs


def minmax_for_plot(values: Sequence[float]) -> np.ndarray:
    """Normalize one episode for plotting only; never use across episodes."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.astype(np.float32)
    low = float(array.min())
    high = float(array.max())
    if math.isclose(low, high):
        return np.ones_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def rank_correlation(values: Sequence[float]) -> float | None:
    """Compute Spearman correlation against chronological order."""

    if len(values) < 2:
        return None
    ranks = pd.Series(values, dtype=np.float64).rank(method="average").to_numpy()
    timeline = np.arange(len(values), dtype=np.float64)
    if np.std(ranks) == 0:
        return 0.0
    return float(np.corrcoef(ranks, timeline)[0, 1])


def classify_collection(dataset_name: str, metadata: dict[str, Any]) -> str:
    if metadata.get("collection_type") == "hil" or metadata.get("hil_protocol"):
        return "hil_correction"
    terminal_success = metadata.get("terminal_success")
    is_policy_rollout = metadata.get("episode_type") == "autonomous_rollout"
    if is_policy_rollout and terminal_success is True:
        return "policy_success"
    if is_policy_rollout and terminal_success is False:
        return "policy_failure"
    return "expert"


def discover_dataset_roots(input_root: Path, selected: set[str] | None = None) -> list[Path]:
    """Find direct child roots containing published Orbit episodes."""

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist or is not a directory: {input_root}")
    roots = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if selected is not None and path.name not in selected:
            continue
        if any(path.glob("episode-*/episode_metadata.json")):
            roots.append(path)
    if selected is not None:
        found = {path.name for path in roots}
        missing = sorted(selected - found)
        if missing:
            raise FileNotFoundError(f"Selected dataset roots not found: {missing}")
    if not roots:
        raise ValueError(f"No Orbit dataset roots found under {input_root}")
    return roots


def decode_selected_frames(video_path: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode selected RGB frames in one sequential pass through a video."""

    wanted = {int(index) for index in indices}
    if not wanted:
        return {}
    if min(wanted) < 0:
        raise ValueError(f"Negative video frame requested from {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        frame_index = 0
        final_index = max(wanted)
        while frame_index <= final_index:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(wanted - set(decoded))
    if missing:
        raise ValueError(f"Could not decode video frame(s) {missing[:10]} from {video_path}")
    return decoded


def single_token_candidate_ids(tokenizer: Any, prompt: str, candidates: Sequence[str]) -> list[int]:
    """Return candidate token ids, requiring one token appended to an identical prompt."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = []
    for candidate in candidates:
        full_ids = tokenizer.encode(f"{prompt}{candidate}", add_special_tokens=False)
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
            raise ValueError(
                f"TOPReward candidate {candidate!r} is not one token in this prompt context; "
                "the shared-logit scorer requires single-token answers."
            )
        candidate_ids.append(int(full_ids[-1]))
    return candidate_ids


def split_video_metadata(video_inputs: Sequence[Any] | None) -> tuple[list[Any] | None, list[Any] | None]:
    """Split qwen-vl-utils ``(video, metadata)`` pairs for the Qwen3 processor."""

    if video_inputs is None:
        return None, None
    videos = []
    metadata = []
    for item in video_inputs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Expected qwen-vl-utils to return (video, metadata) pairs")
        video, video_metadata = item
        videos.append(video)
        metadata.append(video_metadata)
    return videos, metadata


class QwenTOPRewardScorer:
    """Direct-prompt TOPReward scorer using Qwen3-VL token log-probabilities."""

    def __init__(
        self,
        model_name: str,
        *,
        torch_dtype: str,
        attn_implementation: str | None,
        fps: float,
    ) -> None:
        import torch
        from torch.nn import functional
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info

        load_kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": "auto"}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model_name = model_name
        self.fps = fps
        self._torch = torch
        self._functional = functional
        self._process_vision_info = process_vision_info

    def _score_answers(
        self,
        frames: Sequence[np.ndarray],
        text: str,
        answers: Sequence[str],
    ) -> list[float]:
        from PIL import Image

        pil_frames = [Image.fromarray(frame) for frame in frames]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "sample_fps": self.fps,
                        "raw_fps": self.fps,
                    },
                    {"type": "text", "text": text.rstrip()},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        eos_token = self.processor.tokenizer.eos_token
        if eos_token is not None:
            prompt = prompt.split(eos_token)[0]
        prompt = prompt.rstrip()
        candidate_ids = single_token_candidate_ids(self.processor.tokenizer, prompt, answers)
        image_patch_size = int(getattr(self.processor.image_processor, "patch_size", 16))
        image_inputs, video_inputs_with_metadata, video_kwargs = self._process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
            image_patch_size=image_patch_size,
        )
        video_inputs, video_metadata = split_video_metadata(video_inputs_with_metadata)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        next_token_log_probs = self._functional.log_softmax(outputs.logits[0, -1, :], dim=-1)
        return [float(next_token_log_probs[token_id].item()) for token_id in candidate_ids]

    def score_true_false(self, frames: Sequence[np.ndarray], instruction: str) -> tuple[float, float]:
        stem = DEFAULT_PROMPT_PREFIX + DEFAULT_BOOLEAN_SUFFIX.format(instruction=instruction, answer="")
        logp_true, logp_false = self._score_answers(frames, stem, [" True", " False"])
        return logp_true, logp_false

    def score_yes_no(self, frames: Sequence[np.ndarray], instruction: str) -> tuple[float, float]:
        stem = DEFAULT_YES_NO_SUFFIX.format(instruction=instruction, answer="")
        logp_yes, logp_no = self._score_answers(frames, stem, [" Yes", " No"])
        return logp_yes, logp_no


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def score_episode(
    scorer: QwenTOPRewardScorer,
    dataset_root: Path,
    episode_dir: Path,
    *,
    camera: str,
    num_anchors: int,
    max_frames: int,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
    instruction = str(metadata.get("task_description", "")).strip()
    if not instruction:
        raise ValueError(f"Missing task_description: {episode_dir}")
    timesteps = pd.read_parquet(episode_dir / "timesteps.parquet")
    timesteps = timesteps.sort_values("timestep_index").reset_index(drop=True)
    specs = build_prefix_specs(
        timesteps,
        camera=camera,
        num_anchors=num_anchors,
        max_frames=max_frames,
    )
    all_video_indices = [index for spec in specs for index in spec.sampled_video_frame_indices]
    decoded = decode_selected_frames(episode_dir / "videos" / f"{camera}.mp4", all_video_indices)

    anchor_rows: list[dict[str, Any]] = []
    for spec in specs:
        frames = [decoded[index] for index in spec.sampled_video_frame_indices]
        logp_true, logp_false = scorer.score_true_false(frames, instruction)
        anchor_rows.append(
            {
                **asdict(spec),
                "sampled_frame_count": len(frames),
                "logp_true": logp_true,
                "logp_false": logp_false,
                "true_false_margin": logp_true - logp_false,
            }
        )
        logger.info(
            "%s/%s anchor %d/%d: t=%d frames=%d true=%.4f false=%.4f margin=%.4f",
            dataset_root.name,
            episode_dir.name,
            spec.anchor_index + 1,
            len(specs),
            spec.anchor_timestep_index,
            len(frames),
            logp_true,
            logp_false,
            logp_true - logp_false,
        )

    true_plot = minmax_for_plot([row["logp_true"] for row in anchor_rows])
    margin_plot = minmax_for_plot([row["true_false_margin"] for row in anchor_rows])
    for row, true_value, margin_value in zip(anchor_rows, true_plot, margin_plot, strict=True):
        row["normalized_true_for_plot"] = float(true_value)
        row["normalized_margin_for_plot"] = float(margin_value)

    final_spec = specs[-1]
    full_frames = [decoded[index] for index in final_spec.sampled_video_frame_indices]
    logp_yes, logp_no = scorer.score_yes_no(full_frames, instruction)
    margins = np.asarray([row["true_false_margin"] for row in anchor_rows], dtype=np.float64)
    edge_count = min(3, len(margins))
    prefix_margin_change = float(margins[-edge_count:].mean() - margins[:edge_count].mean())
    collection_type = classify_collection(dataset_root.name, metadata)
    terminal_success = metadata.get("terminal_success")
    if terminal_success is None and collection_type in {"policy_success", "policy_failure"}:
        terminal_success = collection_type == "policy_success"
    return {
        "status": "complete",
        "dataset": dataset_root.name,
        "episode_id": episode_dir.name,
        "collection_type": collection_type,
        "terminal_success": terminal_success,
        "metadata_success": metadata.get("success"),
        "task": instruction,
        "num_timesteps": len(timesteps),
        "timestep_indices": [int(value) for value in timesteps["timestep_index"]],
        "timestep_video_frame_indices": [
            None if pd.isna(value) else int(value)
            for value in timesteps[f"{camera}_video_frame_index"]
        ],
        "camera": camera,
        "model_name": scorer.model_name,
        "scoring_config": scoring_config,
        "num_prefix_anchors": len(specs),
        "max_frames_per_prefix": max_frames,
        "anchors": anchor_rows,
        "terminal_logp_true": float(anchor_rows[-1]["logp_true"]),
        "terminal_logp_false": float(anchor_rows[-1]["logp_false"]),
        "terminal_true_false_margin": float(anchor_rows[-1]["true_false_margin"]),
        "prefix_margin_change": prefix_margin_change,
        "full_video_logp_yes": logp_yes,
        "full_video_logp_no": logp_no,
        "full_video_yes_no_margin": logp_yes - logp_no,
        "voc_true": rank_correlation([row["logp_true"] for row in anchor_rows]),
        "voc_margin": rank_correlation([row["true_false_margin"] for row in anchor_rows]),
    }


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=np.float64)
    return (values - float(values.mean())) / std


def aggregate_results(output_dir: Path) -> tuple[Path, Path, Path]:
    """Build episode, anchor, and timestep Parquet files from completed JSON sidecars."""

    payloads = []
    for path in sorted((output_dir / "episodes").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            payloads.append(payload)
    if not payloads:
        raise ValueError(f"No completed episode scores found under {output_dir / 'episodes'}")

    episode_rows = []
    anchor_rows = []
    timestep_rows = []
    for payload in payloads:
        anchors = payload["anchors"]
        episode_rows.append(
            {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "anchors",
                    "scoring_config",
                    "timestep_indices",
                    "timestep_video_frame_indices",
                }
            }
        )
        for anchor in anchors:
            anchor_rows.append(
                {
                    "dataset": payload["dataset"],
                    "episode_id": payload["episode_id"],
                    "collection_type": payload["collection_type"],
                    "terminal_success": payload["terminal_success"],
                    "task": payload["task"],
                    **anchor,
                }
            )
        anchor_times = np.asarray([row["anchor_timestep_index"] for row in anchors], dtype=np.float64)
        timestep_indices = payload["timestep_indices"]
        video_frame_indices = payload["timestep_video_frame_indices"]
        all_times = np.asarray(timestep_indices, dtype=np.float64)
        interpolated = {}
        for key in ("logp_true", "logp_false", "true_false_margin"):
            values = np.asarray([row[key] for row in anchors], dtype=np.float64)
            interpolated[key] = np.interp(all_times, anchor_times, values)
        anchor_by_timestep = {int(row["anchor_timestep_index"]): row["anchor_index"] for row in anchors}
        for position, (timestep_index, video_frame_index) in enumerate(
            zip(timestep_indices, video_frame_indices, strict=True)
        ):
            timestep_rows.append(
                {
                    "dataset": payload["dataset"],
                    "episode_id": payload["episode_id"],
                    "collection_type": payload["collection_type"],
                    "terminal_success": payload["terminal_success"],
                    "task": payload["task"],
                    "timestep_index": timestep_index,
                    "video_frame_index": video_frame_index,
                    "is_anchor": timestep_index in anchor_by_timestep,
                    "anchor_index": anchor_by_timestep.get(timestep_index),
                    "topreward_logp_true": float(interpolated["logp_true"][position]),
                    "topreward_logp_false": float(interpolated["logp_false"][position]),
                    "topreward_true_false_margin": float(interpolated["true_false_margin"][position]),
                }
            )

    episodes = pd.DataFrame(episode_rows)
    policy_mask = episodes["collection_type"].isin(["policy_success", "policy_failure"])
    episodes["success_score"] = np.nan
    if bool(policy_mask.any()):
        prefix_z = _zscore(episodes.loc[policy_mask, "prefix_margin_change"])
        yes_no_z = _zscore(episodes.loc[policy_mask, "full_video_yes_no_margin"])
        episodes.loc[policy_mask, "success_score"] = prefix_z + yes_no_z

    episode_path = output_dir / "episode_scores.parquet"
    anchor_path = output_dir / "anchor_scores.parquet"
    timestep_path = output_dir / "timestep_scores.parquet"
    episodes.to_parquet(episode_path, index=False)
    pd.DataFrame(anchor_rows).to_parquet(anchor_path, index=False)
    pd.DataFrame(timestep_rows).to_parquet(timestep_path, index=False)

    grouped = []
    for collection_type, frame in episodes.groupby("collection_type"):
        grouped.append(
            {
                "collection_type": collection_type,
                "episodes": len(frame),
                "mean_terminal_logp_true": float(frame["terminal_logp_true"].mean()),
                "mean_terminal_true_false_margin": float(frame["terminal_true_false_margin"].mean()),
                "mean_voc_true": float(frame["voc_true"].mean()),
                "mean_voc_margin": float(frame["voc_margin"].mean()),
            }
        )
    atomic_write_json(output_dir / "summary.json", {"by_collection_type": grouped})
    return episode_path, anchor_path, timestep_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset child name; repeatable.")
    parser.add_argument("--episode", action="append", dest="episodes", help="Episode id; repeatable.")
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--num-anchors", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=2.0, help="Video FPS metadata passed to Qwen3-VL.")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip completed per-episode sidecars.")
    parser.add_argument("--overwrite", action="store_true", help="Replace completed per-episode sidecars.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scoring_config = {
        "scorer_version": SCORER_VERSION,
        "input_root": str(input_root),
        "model_name": args.model_name,
        "camera": args.camera,
        "num_prefix_anchors": args.num_anchors,
        "max_frames_per_prefix": args.max_frames,
        "fps": args.fps,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "boolean_prompt_suffix": DEFAULT_BOOLEAN_SUFFIX,
        "yes_no_prompt_suffix": DEFAULT_YES_NO_SUFFIX,
        "normalization_scope": "within_episode_plots_only",
    }
    config_path = output_dir / "config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != scoring_config:
            if not args.overwrite:
                raise ValueError(
                    f"Existing scoring configuration differs: {config_path}. "
                    "Use the original arguments or pass --overwrite to replace all scores."
                )
            shutil.rmtree(output_dir / "episodes", ignore_errors=True)

    dataset_roots = discover_dataset_roots(input_root, set(args.datasets) if args.datasets else None)
    selected_episodes = set(args.episodes) if args.episodes else None
    jobs: list[tuple[Path, Path, Path]] = []
    completed_count = 0
    discovered_episode_names: set[str] = set()
    for dataset_root in dataset_roots:
        for episode_dir in sorted(dataset_root.glob("episode-*")):
            if selected_episodes is not None and episode_dir.name not in selected_episodes:
                continue
            discovered_episode_names.add(episode_dir.name)
            output_path = output_dir / "episodes" / dataset_root.name / f"{episode_dir.name}.json"
            if output_path.exists() and args.resume:
                try:
                    if json.loads(output_path.read_text(encoding="utf-8")).get("status") == "complete":
                        logger.info("Skipping completed %s/%s", dataset_root.name, episode_dir.name)
                        completed_count += 1
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {output_path}. Pass --resume or --overwrite.")
            jobs.append((dataset_root, episode_dir, output_path))
    if selected_episodes is not None:
        missing_episodes = selected_episodes - discovered_episode_names
        if missing_episodes:
            raise FileNotFoundError(f"Selected episodes not found: {sorted(missing_episodes)}")

    atomic_write_json(config_path, scoring_config)

    if jobs:
        logger.info("Loading %s for %d episode(s)", args.model_name, len(jobs))
        scorer = QwenTOPRewardScorer(
            args.model_name,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            fps=args.fps,
        )
        total_count = completed_count + len(jobs)
        with tqdm(
            total=total_count,
            initial=completed_count,
            desc="TOPReward episodes",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            for dataset_root, episode_dir, output_path in jobs:
                progress.set_postfix_str(f"{dataset_root.name}/{episode_dir.name}")
                try:
                    result = score_episode(
                        scorer,
                        dataset_root,
                        episode_dir,
                        camera=args.camera,
                        num_anchors=args.num_anchors,
                        max_frames=args.max_frames,
                        scoring_config=scoring_config,
                    )
                    atomic_write_json(output_path, result)
                except Exception as exc:
                    logger.exception("Failed %s/%s", dataset_root.name, episode_dir.name)
                    atomic_write_json(
                        output_path.with_suffix(".error.json"),
                        {
                            "status": "error",
                            "dataset": dataset_root.name,
                            "episode_id": episode_dir.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    if args.fail_fast:
                        raise
                finally:
                    progress.update(1)
    else:
        logger.info("No episodes require scoring; rebuilding aggregate outputs")

    paths = aggregate_results(output_dir)
    logger.info("Wrote aggregate results: %s", ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
