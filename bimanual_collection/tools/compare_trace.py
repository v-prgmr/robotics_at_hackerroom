"""Compare a bimanual inference debug trace against an intermediate dataset episode."""

from __future__ import annotations

import argparse
import html
import json
import logging
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceData:
    metadata: dict[str, Any]
    observations: list[dict[str, Any]]
    chunks_by_sequence: dict[int, dict[str, Any]]
    robot_actions: list[dict[str, Any]]
    state_transitions: list[dict[str, Any]]
    holds: list[dict[str, Any]]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_trace(trace_dir: Path) -> TraceData:
    metadata_path = trace_dir / "metadata.json"
    events_path = trace_dir / "events.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing trace metadata: {metadata_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing trace events: {events_path}")

    metadata = read_json(metadata_path)
    observations: list[dict[str, Any]] = []
    chunks_by_sequence: dict[int, dict[str, Any]] = {}
    robot_actions: list[dict[str, Any]] = []
    state_transitions: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("event")
            if kind == "policy_observation":
                observations.append(event)
            elif kind == "policy_chunk":
                chunks_by_sequence[int(event["observation_sequence"])] = event
            elif kind == "robot_action":
                robot_actions.append(event)
            elif kind == "state_transition":
                state_transitions.append(event)
            elif kind == "hold":
                holds.append(event)
            else:
                logger.debug("Ignoring trace event at line %d: %s", line_number, kind)

    observations.sort(key=lambda event: int(event.get("observation_sequence", 0)))
    return TraceData(metadata, observations, chunks_by_sequence, robot_actions, state_transitions, holds)


def load_episode(episode_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata_path = episode_dir / "episode_metadata.json"
    timesteps_path = episode_dir / "timesteps.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing episode metadata: {metadata_path}")
    if not timesteps_path.exists():
        raise FileNotFoundError(f"Missing episode timesteps: {timesteps_path}")
    return read_json(metadata_path), pd.read_parquet(timesteps_path)


def vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def dataset_state_vector(row: pd.Series) -> np.ndarray:
    return np.concatenate([vector(row["left_follower_joints"]), vector(row["right_follower_joints"])]).astype(
        np.float32,
        copy=False,
    )


def dataset_action_vector(row: pd.Series) -> np.ndarray:
    return np.concatenate([vector(row["left_commanded_action"]), vector(row["right_commanded_action"])]).astype(
        np.float32,
        copy=False,
    )


def metric_row(prefix: str, trace_vector: np.ndarray | None, dataset_vector: np.ndarray | None) -> dict[str, Any]:
    if trace_vector is None or dataset_vector is None:
        return {
            f"{prefix}_l2": None,
            f"{prefix}_mean_abs": None,
            f"{prefix}_max_abs": None,
            f"{prefix}_dim_match": False,
        }
    if trace_vector.shape != dataset_vector.shape:
        return {
            f"{prefix}_l2": None,
            f"{prefix}_mean_abs": None,
            f"{prefix}_max_abs": None,
            f"{prefix}_dim_match": False,
        }
    diff = trace_vector - dataset_vector
    return {
        f"{prefix}_l2": float(np.linalg.norm(diff)),
        f"{prefix}_mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        f"{prefix}_max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        f"{prefix}_dim_match": True,
    }


def observation_by_source_timestamp(trace: TraceData, timestamp_s: float | None) -> dict[str, Any] | None:
    if timestamp_s is None or not trace.observations:
        return None
    return min(
        trace.observations,
        key=lambda event: abs(float(event.get("observation_timestamp_s", 0.0)) - float(timestamp_s)),
    )


def compare_vectors(
    trace: TraceData,
    episode_df: pd.DataFrame,
    *,
    trace_start_index: int = 0,
    dataset_start_index: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    available = min(len(trace.robot_actions) - trace_start_index, len(episode_df) - dataset_start_index)
    if available < 0:
        available = 0
    count = available if limit is None else min(available, limit)
    rows: list[dict[str, Any]] = []
    for offset in range(count):
        trace_index = trace_start_index + offset
        dataset_index = dataset_start_index + offset
        trace_action = trace.robot_actions[trace_index]
        dataset_row = episode_df.iloc[dataset_index]
        source_observation = observation_by_source_timestamp(trace, trace_action.get("source_timestamp_s"))
        trace_state = vector(source_observation["observation_state"]) if source_observation is not None else None
        trace_action_vector = vector(trace_action["action"])
        dataset_state = dataset_state_vector(dataset_row)
        dataset_action = dataset_action_vector(dataset_row)
        row: dict[str, Any] = {
            "comparison_index": offset,
            "trace_action_index": trace_index,
            "dataset_row_position": dataset_index,
            "dataset_timestep_index": int(dataset_row.get("timestep_index", dataset_index)),
            "trace_observation_sequence": None
            if source_observation is None
            else int(source_observation["observation_sequence"]),
            "trace_source_timestamp_s": trace_action.get("source_timestamp_s"),
            "dataset_monotonic_timestamp_s": dataset_row.get("monotonic_timestamp_s"),
        }
        row.update(metric_row("state", trace_state, dataset_state))
        row.update(metric_row("action", trace_action_vector, dataset_action))
        row.update(metric_row("left_action", trace_action_vector[:6], dataset_action[:6]))
        row.update(metric_row("right_action", trace_action_vector[6:], dataset_action[6:]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df or df.empty:
        return {"mean": None, "max": None, "argmax_comparison_index": None}
    values = pd.Series(pd.to_numeric(df[column], errors="coerce")).dropna()
    if values.empty:
        return {"mean": None, "max": None, "argmax_comparison_index": None}
    max_index = int(str(values.idxmax()))
    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "argmax_comparison_index": int(df.loc[max_index, "comparison_index"]),
    }


def build_report(trace: TraceData, episode_metadata: dict[str, Any], episode_df: pd.DataFrame, vector_df: pd.DataFrame) -> dict[str, Any]:
    trace_task = trace.metadata.get("task_description")
    episode_task = episode_metadata.get("task_description")
    return {
        "trace": {
            "task_description": trace_task,
            "policy_observations": len(trace.observations),
            "policy_chunks": len(trace.chunks_by_sequence),
            "robot_actions": len(trace.robot_actions),
            "holds": len(trace.holds),
            "state_transitions": [event.get("state") for event in trace.state_transitions],
        },
        "episode": {
            "episode_id": episode_metadata.get("episode_id"),
            "task_description": episode_task,
            "timesteps": int(len(episode_df)),
        },
        "task_match": trace_task == episode_task,
        "compared_steps": int(len(vector_df)),
        "metrics": {
            "state_mean_abs": summarize_metric(vector_df, "state_mean_abs"),
            "state_max_abs": summarize_metric(vector_df, "state_max_abs"),
            "action_mean_abs": summarize_metric(vector_df, "action_mean_abs"),
            "action_max_abs": summarize_metric(vector_df, "action_max_abs"),
            "left_action_mean_abs": summarize_metric(vector_df, "left_action_mean_abs"),
            "right_action_mean_abs": summarize_metric(vector_df, "right_action_mean_abs"),
        },
    }


def camera_names_from_episode(df: pd.DataFrame) -> list[str]:
    return sorted(column.removesuffix("_video_frame_index") for column in df.columns if column.endswith("_video_frame_index"))


def trace_observation_by_sequence(trace: TraceData) -> dict[int, dict[str, Any]]:
    return {int(event["observation_sequence"]): event for event in trace.observations}


def compare_images(
    trace_dir: Path,
    trace: TraceData,
    episode_dir: Path,
    episode_df: pd.DataFrame,
    vector_df: pd.DataFrame,
    *,
    image_limit: int | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    import cv2  # type: ignore

    from bimanual_collection.recording.backends.lerobot_export import SequentialVideoFrameReader

    observation_by_sequence = trace_observation_by_sequence(trace)
    cameras = camera_names_from_episode(episode_df)
    readers: dict[str, SequentialVideoFrameReader] = {}
    visuals_dir = None
    if output_dir is not None:
        visuals_dir = output_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        comparison_rows = vector_df if image_limit is None else vector_df.head(image_limit)
        for _, comparison in comparison_rows.iterrows():
            sequence = comparison.get("trace_observation_sequence")
            if sequence is None or pd.isna(sequence):
                continue
            observation = observation_by_sequence.get(int(sequence))
            if observation is None:
                continue
            trace_cameras = {camera["camera_name"]: camera for camera in observation.get("cameras", [])}
            dataset_index = int(comparison["dataset_row_position"])
            dataset_row = episode_df.iloc[dataset_index]
            for camera_name in cameras:
                trace_camera = trace_cameras.get(camera_name)
                trace_image_path = None if trace_camera is None else trace_camera.get("image_path")
                dataset_video_index = dataset_row.get(f"{camera_name}_video_frame_index")
                if not trace_image_path or dataset_video_index is None or bool(pd.isna(dataset_video_index)):
                    rows.append(
                        {
                            "comparison_index": int(comparison["comparison_index"]),
                            "camera_name": camera_name,
                            "trace_image_path": trace_image_path,
                            "dataset_video_frame_index": dataset_video_index,
                            "mean_abs_rgb": None,
                            "max_abs_rgb": None,
                            "shape_match": False,
                            "visual_path": None,
                            "error": "missing trace image or dataset video frame",
                        }
                    )
                    continue
                video_path = episode_dir / "videos" / f"{camera_name}.mp4"
                reader = readers.get(camera_name)
                if reader is None:
                    reader = SequentialVideoFrameReader(video_path)
                    readers[camera_name] = reader
                trace_image_bgr = cv2.imread(str(trace_dir / trace_image_path), cv2.IMREAD_COLOR)
                if trace_image_bgr is None:
                    raise ValueError(f"Could not read trace image: {trace_dir / trace_image_path}")
                trace_image = cv2.cvtColor(trace_image_bgr, cv2.COLOR_BGR2RGB)
                dataset_image = reader.get(int(dataset_video_index))
                shape_match = trace_image.shape == dataset_image.shape
                if not shape_match:
                    dataset_image = cv2.resize(dataset_image, (trace_image.shape[1], trace_image.shape[0]))
                diff = trace_image.astype(np.float32) - dataset_image.astype(np.float32)
                visual_path = None
                if visuals_dir is not None:
                    diff_image = np.clip(np.abs(diff), 0, 255).astype(np.uint8)
                    panel = np.concatenate([trace_image, dataset_image, diff_image], axis=1)
                    visual_path = visuals_dir / f"compare_{int(comparison['comparison_index']):06d}_{camera_name}.png"
                    cv2.imwrite(str(visual_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                rows.append(
                    {
                        "comparison_index": int(comparison["comparison_index"]),
                        "camera_name": camera_name,
                        "trace_image_path": trace_image_path,
                        "dataset_video_frame_index": int(dataset_video_index),
                        "mean_abs_rgb": float(np.mean(np.abs(diff))),
                        "max_abs_rgb": float(np.max(np.abs(diff))),
                        "shape_match": bool(shape_match),
                        "visual_path": str(visual_path.relative_to(output_dir)) if visual_path is not None and output_dir is not None else None,
                        "error": None,
                    }
                )
    finally:
        for reader in readers.values():
            reader.close()
    return pd.DataFrame(rows)


def html_metric_table(report: dict[str, Any]) -> str:
    rows = []
    for name, metric in report["metrics"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(metric['mean']))}</td>"
            f"<td>{html.escape(str(metric['max']))}</td>"
            f"<td>{html.escape(str(metric['argmax_comparison_index']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def html_worst_rows(df: pd.DataFrame, column: str, limit: int = 10) -> str:
    if column not in df or df.empty:
        return "<p>No rows.</p>"
    table = df.copy()
    table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values(column, ascending=False).head(limit)
    rows = []
    for _, row in table.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(row['comparison_index'])}</td>"
            f"<td>{int(row['trace_action_index'])}</td>"
            f"<td>{int(row['dataset_timestep_index'])}</td>"
            f"<td>{html.escape(str(row.get(column)))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Comparison</th><th>Trace Action</th><th>Dataset Timestep</th>"
        f"<th>{html.escape(column)}</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def html_camera_panels(image_df: pd.DataFrame | None, limit: int = 60) -> str:
    if image_df is None or image_df.empty or "visual_path" not in image_df:
        return "<p>No camera visual panels. Run with <code>--compare-images</code> to generate them.</p>"
    rows = []
    for _, row in image_df.head(limit).iterrows():
        visual_path = row.get("visual_path")
        if not visual_path or pd.isna(visual_path):
            continue
        rows.append(
            "<figure>"
            f"<img src='{html.escape(str(visual_path))}' alt='camera comparison'>"
            "<figcaption>"
            f"comparison {int(row['comparison_index'])}, camera {html.escape(str(row['camera_name']))}, "
            f"mean_abs_rgb={html.escape(str(row.get('mean_abs_rgb')))}"
            "<br><span>left: trace, middle: dataset, right: absolute RGB diff</span>"
            "</figcaption>"
            "</figure>"
        )
    return "".join(rows) or "<p>No camera visual panels were generated.</p>"


def write_html_report(
    output_dir: Path,
    report: dict[str, Any],
    vector_df: pd.DataFrame,
    image_df: pd.DataFrame | None,
) -> Path:
    index_path = output_dir / "index.html"
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bimanual Trace Comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #c8d0dc; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #c8d0dc; }}
    figure {{ margin: 0 0 24px; }}
    figcaption {{ font-size: 0.9rem; color: #475569; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Bimanual Trace Comparison</h1>
  <p><strong>Task match:</strong> {html.escape(str(report['task_match']))}</p>
  <p><strong>Compared steps:</strong> {html.escape(str(report['compared_steps']))}</p>
  <p><a href="report.json">report.json</a> | <a href="vector_comparison.csv">vector_comparison.csv</a> | <a href="camera_comparison.csv">camera_comparison.csv</a></p>
  <h2>Summary Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Mean</th><th>Max</th><th>Argmax Step</th></tr></thead><tbody>{html_metric_table(report)}</tbody></table>
  <h2>Worst Action Differences</h2>
  {html_worst_rows(vector_df, 'action_mean_abs')}
  <h2>Worst State Differences</h2>
  {html_worst_rows(vector_df, 'state_mean_abs')}
  <h2>Camera Panels</h2>
  {html_camera_panels(image_df)}
</body>
</html>
"""
    index_path.write_text(content, encoding="utf-8")
    return index_path


def write_outputs(output_dir: Path, report: dict[str, Any], vector_df: pd.DataFrame, image_df: pd.DataFrame | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    vector_df.to_csv(output_dir / "vector_comparison.csv", index=False)
    if image_df is not None:
        image_df.to_csv(output_dir / "camera_comparison.csv", index=False)
    write_html_report(output_dir, report, vector_df, image_df)


def print_summary(report: dict[str, Any], output_dir: Path | None) -> None:
    print("Trace vs episode comparison")
    print(f"  Trace observations: {report['trace']['policy_observations']}")
    print(f"  Trace robot actions: {report['trace']['robot_actions']}")
    print(f"  Episode timesteps: {report['episode']['timesteps']}")
    print(f"  Compared steps: {report['compared_steps']}")
    print(f"  Task match: {report['task_match']}")
    for name, metric in report["metrics"].items():
        print(f"  {name}: mean={metric['mean']} max={metric['max']} argmax={metric['argmax_comparison_index']}")
    if output_dir is not None:
        print(f"Wrote comparison files to: {output_dir}")
        print(f"Open HTML report: {output_dir / 'index.html'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True, help="Debug trace run directory.")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Intermediate episode-* directory.")
    parser.add_argument("--output-dir", type=Path, help="Directory for report.json and CSV outputs.")
    parser.add_argument("--trace-start-index", type=int, default=0, help="First trace robot_action index to compare.")
    parser.add_argument("--dataset-start-index", type=int, default=0, help="First dataset timestep index to compare.")
    parser.add_argument("--limit", type=int, help="Maximum number of aligned steps to compare.")
    parser.add_argument("--compare-images", action="store_true", help="Compare trace camera PNGs to episode video frames.")
    parser.add_argument("--image-limit", type=int, help="Maximum number of aligned steps for image comparison.")
    parser.add_argument("--open-report", action="store_true", help="Open output index.html in the default browser after writing it.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    trace_dir = args.trace_dir.expanduser()
    episode_dir = args.episode_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None

    if args.trace_start_index < 0:
        parser.error("--trace-start-index must be >= 0")
    if args.dataset_start_index < 0:
        parser.error("--dataset-start-index must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.image_limit is not None and args.image_limit < 1:
        parser.error("--image-limit must be >= 1")
    if args.open_report and output_dir is None:
        parser.error("--open-report requires --output-dir")

    trace = load_trace(trace_dir)
    episode_metadata, episode_df = load_episode(episode_dir)
    vector_df = compare_vectors(
        trace,
        episode_df,
        trace_start_index=int(args.trace_start_index),
        dataset_start_index=int(args.dataset_start_index),
        limit=args.limit,
    )
    if vector_df.empty:
        parser.error("No overlapping trace robot actions and dataset timesteps to compare")
    image_df = None
    if args.compare_images:
        image_df = compare_images(
            trace_dir,
            trace,
            episode_dir,
            episode_df,
            vector_df,
            image_limit=args.image_limit,
            output_dir=output_dir,
        )
    report = build_report(trace, episode_metadata, episode_df, vector_df)
    if image_df is not None:
        report["image_comparisons"] = int(len(image_df))
        report["metrics"]["camera_mean_abs_rgb"] = summarize_metric(image_df, "mean_abs_rgb")
    if output_dir is not None:
        write_outputs(output_dir, report, vector_df, image_df)
        if args.open_report:
            webbrowser.open((output_dir / "index.html").resolve().as_uri())
    print_summary(report, output_dir)


if __name__ == "__main__":
    main()
