"""Replay Orbit intermediate dataset episodes directly into Rerun."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from bimanual_collection.recording.backends.lerobot_export import SequentialVideoFrameReader

logger = logging.getLogger(__name__)
CLEAR_ROOTS = (
    "cameras",
    "followers",
    "leaders",
    "actions",
    "topreward",
    "episode_metadata",
    "episodes",
)

TOPREWARD_COLUMNS = (
    "topreward_logp_true",
    "topreward_logp_false",
    "topreward_true_false_margin",
)
TIME_TIMELINES = ("episode_time", "monotonic_time", "wall_time")


@dataclass(frozen=True)
class TopRewardResults:
    timestep_by_episode: dict[str, pd.DataFrame]
    episode_by_id: dict[str, dict[str, Any]]


def load_topreward_results(path: Path, *, dataset_name: str) -> TopRewardResults:
    """Load and validate TOPReward aggregates for one Orbit dataset root."""

    path = path.expanduser()
    score_dir = path if path.is_dir() else path.parent
    timestep_path = path / "timestep_scores.parquet" if path.is_dir() else path
    if not timestep_path.exists():
        raise FileNotFoundError(f"TOPReward timestep scores not found: {timestep_path}")

    scores = cast(pd.DataFrame, pd.read_parquet(timestep_path))
    required = {"dataset", "episode_id", "timestep_index", "is_anchor", *TOPREWARD_COLUMNS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"TOPReward timestep scores are missing columns: {sorted(missing)}")
    scores = scores[scores["dataset"].astype(str) == dataset_name].copy()
    if scores.empty:
        raise ValueError(f"No TOPReward rows found for dataset {dataset_name!r} in {timestep_path}")
    if bool(scores.duplicated(subset=["episode_id", "timestep_index"]).any()):
        raise ValueError(f"Duplicate TOPReward episode/timestep rows found in {timestep_path}")

    scores["episode_id"] = scores["episode_id"].astype(str)
    scores["timestep_index"] = np.asarray(
        pd.to_numeric(scores["timestep_index"], errors="raise"), dtype=np.int64
    )
    timestep_by_episode: dict[str, pd.DataFrame] = {}
    for episode_id, episode_scores in scores.groupby("episode_id", sort=False):
        episode_scores = episode_scores.sort_values(by="timestep_index").set_index("timestep_index")
        episode_scores["topreward_true_normalized_plot"] = _minmax_for_plot(
            episode_scores["topreward_logp_true"]
        )
        episode_scores["topreward_margin_normalized_plot"] = _minmax_for_plot(
            episode_scores["topreward_true_false_margin"]
        )
        timestep_by_episode[str(episode_id)] = episode_scores

    episode_by_id: dict[str, dict[str, Any]] = {}
    episode_path = score_dir / "episode_scores.parquet"
    if episode_path.exists():
        episodes = cast(pd.DataFrame, pd.read_parquet(episode_path))
        episode_required = {"dataset", "episode_id"}
        episode_missing = episode_required - set(episodes.columns)
        if episode_missing:
            raise ValueError(f"TOPReward episode scores are missing columns: {sorted(episode_missing)}")
        episodes = episodes[episodes["dataset"].astype(str) == dataset_name]
        if bool(episodes.duplicated(subset=["episode_id"]).any()):
            raise ValueError(f"Duplicate TOPReward episode rows found in {episode_path}")
        episode_by_id = {
            str(row["episode_id"]): row.to_dict()
            for _index, row in episodes.iterrows()
        }
    return TopRewardResults(timestep_by_episode=timestep_by_episode, episode_by_id=episode_by_id)


def _minmax_for_plot(values: Any) -> np.ndarray:
    array = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=np.float64)
    finite = np.isfinite(array)
    normalized = np.full(array.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return normalized
    low = float(array[finite].min())
    high = float(array[finite].max())
    if math.isclose(low, high):
        normalized[finite] = 1.0
    else:
        normalized[finite] = (array[finite] - low) / (high - low)
    return normalized


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def rerun_entity_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def discover_episode_dirs(
    dataset: Path,
    *,
    episodes: list[str] | None = None,
    max_episodes: int | None = None,
) -> list[Path]:
    dataset = dataset.expanduser()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset}")

    all_episodes = sorted(
        path
        for path in dataset.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "timesteps.parquet").exists()
    )
    if episodes:
        by_name = {path.name: path for path in all_episodes}
        selected: list[Path] = []
        for value in episodes:
            name = normalize_episode_selector(value)
            path = by_name.get(name)
            if path is None:
                raise FileNotFoundError(f"Episode {value!r} was not found under {dataset}")
            selected.append(path)
        all_episodes = selected

    if max_episodes is not None:
        all_episodes = all_episodes[:max_episodes]
    return all_episodes


def normalize_episode_selector(value: str) -> str:
    value = str(value).strip()
    if value.startswith("episode-"):
        return value
    if value.isdigit():
        return f"episode-{int(value):06d}"
    return value


def camera_names_from_episode(
    episode: Path,
    df: pd.DataFrame,
    requested: set[str] | None = None,
) -> list[str]:
    suffix = "_video_frame_index"
    cameras = []
    for column in df.columns:
        if not column.endswith(suffix):
            continue
        camera = column[: -len(suffix)]
        if requested is not None and camera not in requested:
            continue
        if (episode / "videos" / f"{camera}.mp4").exists():
            cameras.append(camera)
    if requested is not None:
        missing = sorted(requested - set(cameras))
        if missing:
            raise FileNotFoundError(f"Requested camera video(s) missing in {episode.name}: {', '.join(missing)}")
    return sorted(cameras)


def replay_episode(
    rr: Any,
    episode: Path,
    *,
    episode_index: int,
    namespace: str = "",
    cameras: set[str] | None = None,
    frame_stride: int = 1,
    frame_limit: int | None = None,
    topreward_results: TopRewardResults | None = None,
) -> int:
    df = pd.read_parquet(episode / "timesteps.parquet")
    metadata = load_json(episode / "episode_metadata.json")
    camera_names = camera_names_from_episode(episode, df, requested=cameras)
    topreward_scores = (
        topreward_results.timestep_by_episode.get(episode.name) if topreward_results is not None else None
    )
    readers = {camera: SequentialVideoFrameReader(episode / "videos" / f"{camera}.mp4") for camera in camera_names}
    logged = 0
    try:
        disable_time_timelines(rr)
        rr.set_time_sequence("episode", episode_index)
        rr.set_time_sequence("timestep", 0)
        _log_episode_metadata(rr, metadata, namespace=namespace, episode=episode)
        start_time_s = _first_float(df, "monotonic_timestamp_s")
        for row_offset, (_row_index, row) in enumerate(df.iterrows()):
            if row_offset % frame_stride != 0:
                continue
            if frame_limit is not None and logged >= frame_limit:
                break
            timestep = _int_or_none(row.get("timestep_index"))
            if timestep is None:
                timestep = row_offset
            rr.set_time_sequence("episode", episode_index)
            rr.set_time_sequence("timestep", timestep)
            timestamp_s = _float_or_none(row.get("monotonic_timestamp_s"))
            if timestamp_s is not None:
                rr.set_time_seconds("episode_time", timestamp_s - start_time_s)
                rr.set_time_seconds("monotonic_time", timestamp_s)
            wall_s = _float_or_none(row.get("wall_timestamp_s"))
            if wall_s is not None:
                rr.set_time_seconds("wall_time", wall_s)

            for camera in camera_names:
                video_index = _int_or_none(row.get(f"{camera}_video_frame_index"))
                if video_index is None:
                    continue
                image = readers[camera].get(video_index)
                rr.log(f"{namespace}cameras/{rerun_entity_name(camera)}/image", rr.Image(image))

            _log_vector(rr, f"{namespace}followers/left", row.get("left_follower_joint_names"), row.get("left_follower_joints"))
            _log_vector(rr, f"{namespace}followers/right", row.get("right_follower_joint_names"), row.get("right_follower_joints"))
            _log_vector(rr, f"{namespace}leaders/left", row.get("left_leader_joint_names"), row.get("left_leader_joints"))
            _log_vector(rr, f"{namespace}leaders/right", row.get("right_leader_joint_names"), row.get("right_leader_joints"))
            _log_vector(rr, f"{namespace}actions/left", row.get("left_action_names"), row.get("left_commanded_action"))
            _log_vector(rr, f"{namespace}actions/right", row.get("right_action_names"), row.get("right_commanded_action"))
            if topreward_scores is not None and timestep in topreward_scores.index:
                _log_topreward_scores(
                    rr,
                    namespace=f"{namespace}topreward/",
                    row=topreward_scores.loc[timestep],
                )
            logged += 1
    finally:
        for reader in readers.values():
            reader.close()
    return logged


def clear_rerun_recording(rr: Any) -> None:
    """Clear previously logged entities when supported by the installed Rerun SDK."""

    clear = getattr(rr, "Clear", None)
    if clear is None:
        return
    for root in CLEAR_ROOTS:
        try:
            rr.log(root, clear(recursive=True))
        except TypeError:
            rr.log(root, clear())


def clear_rerun_timesteps(rr: Any, timesteps: set[int]) -> None:
    """Clear entities at previously-used lazy replay timesteps."""

    clear = getattr(rr, "Clear", None)
    if clear is None:
        return
    disable_time_timelines(rr)
    for timestep in sorted(timesteps):
        rr.set_time_sequence("episode", 0)
        rr.set_time_sequence("timestep", int(timestep))
        for root in CLEAR_ROOTS:
            try:
                rr.log(root, clear(recursive=True))
            except TypeError:
                rr.log(root, clear())


def disable_time_timelines(rr: Any) -> None:
    """Remove stale timestamp context before lazy clear or episode metadata logging."""

    disable = getattr(rr, "disable_timeline", None)
    if disable is None:
        return
    for timeline in TIME_TIMELINES:
        disable(timeline)


def replayed_timesteps(episode: Path, *, frame_stride: int = 1, frame_limit: int | None = None) -> set[int]:
    df = pd.read_parquet(episode / "timesteps.parquet", columns=["timestep_index"])
    timesteps: set[int] = set()
    for row_offset, value in enumerate(df["timestep_index"]):
        if row_offset % frame_stride != 0:
            continue
        if frame_limit is not None and len(timesteps) >= frame_limit:
            break
        timestep = _int_or_none(value)
        timesteps.add(row_offset if timestep is None else timestep)
    return timesteps


def episode_summary(
    episode: Path,
    *,
    index: int,
    total: int,
    topreward_results: TopRewardResults | None = None,
) -> str:
    metadata = load_json(episode / "episode_metadata.json")
    fields = [f"Episode {index + 1} of {total}", episode.name]
    protocol = metadata.get("hil_protocol")
    intervention = metadata.get("intervention_index")
    prior = metadata.get("prior_human_intervention")
    timesteps = metadata.get("num_timesteps")
    if protocol is not None:
        fields.append(f"Protocol: {protocol}")
    if intervention is not None:
        fields.append(f"Intervention: {intervention}")
    if prior is not None:
        fields.append(f"Prior human intervention: {prior}")
    if timesteps is not None:
        fields.append(f"Timesteps: {timesteps}")
    task = str(metadata.get("task_description", "")).strip()
    if task:
        fields.append(f"Task: {task}")
    if topreward_results is not None:
        score_summary = topreward_results.episode_by_id.get(episode.name)
        if score_summary is None:
            status = "timestep scores available" if episode.name in topreward_results.timestep_by_episode else "not scored"
            fields.append(f"TOPReward: {status}")
        else:
            collection_type = score_summary.get("collection_type")
            if collection_type:
                fields.append(f"TOPReward collection: {collection_type}")
            terminal_true = _float_or_none(score_summary.get("terminal_logp_true"))
            terminal_margin = _float_or_none(score_summary.get("terminal_true_false_margin"))
            success_score = _float_or_none(score_summary.get("success_score"))
            if terminal_true is not None:
                fields.append(f"Terminal log P(True): {terminal_true:.4f}")
            if terminal_margin is not None:
                fields.append(f"Terminal True/False margin: {terminal_margin:.4f}")
            if success_score is not None:
                fields.append(f"Policy success score: {success_score:.4f}")
    return "\n".join(fields)


class LazyEpisodeNavigator:
    """Small companion UI for loading one intermediate episode at a time."""

    def __init__(
        self,
        rr: Any,
        episode_dirs: list[Path],
        *,
        application_id: str,
        rerun_connect_grpc: str | None,
        rerun_save: Path | None,
        reconnect_to_spawned_viewer: bool,
        cameras: set[str] | None,
        frame_stride: int,
        frame_limit: int | None,
        topreward_results: TopRewardResults | None,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError("Tkinter is required for --lazy navigation") from exc

        self.rr = rr
        self.episode_dirs = episode_dirs
        self.application_id = application_id
        self.rerun_connect_grpc = rerun_connect_grpc
        self.rerun_save = rerun_save
        self.reconnect_to_spawned_viewer = reconnect_to_spawned_viewer
        self.cameras = cameras
        self.frame_stride = frame_stride
        self.frame_limit = frame_limit
        self.topreward_results = topreward_results
        self.index = 0
        self.previous_timesteps: set[int] = set()
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Orbit Dataset Rerun Navigator")
        self.status = tk.StringVar(value="Ready")
        self.summary = tk.StringVar(value="")

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        ttk.Label(frame, textvariable=self.summary, justify="left", wraplength=520).grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12)
        )
        self.previous_button = ttk.Button(frame, text="Previous", command=self.previous)
        self.previous_button.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.reload_button = ttk.Button(frame, text="Reload", command=self.reload)
        self.reload_button.grid(row=1, column=1, sticky="ew", padx=6)
        self.next_button = ttk.Button(frame, text="Next", command=self.next)
        self.next_button.grid(row=1, column=2, sticky="ew", padx=(6, 0))
        ttk.Label(frame, textvariable=self.status, justify="left").grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

        self.root.bind("<Left>", lambda _event: self.previous())
        self.root.bind("<Right>", lambda _event: self.next())
        self.root.bind("r", lambda _event: self.reload())
        self.root.bind("q", lambda _event: self.root.destroy())
        self._update_buttons()
        self.root.after(50, self.reload)

    def run(self) -> None:
        self.root.mainloop()

    def previous(self) -> None:
        if self.index <= 0:
            return
        self.index -= 1
        self.reload()

    def next(self) -> None:
        if self.index >= len(self.episode_dirs) - 1:
            return
        self.index += 1
        self.reload()

    def reload(self) -> None:
        self._set_busy(True)
        self.root.update_idletasks()
        episode = self.episode_dirs[self.index]
        self.summary.set(
            episode_summary(
                episode,
                index=self.index,
                total=len(self.episode_dirs),
                topreward_results=self.topreward_results,
            )
        )
        self.status.set(f"Loading {episode.name}...")
        self.root.update_idletasks()
        try:
            clear_rerun_timesteps(self.rr, self.previous_timesteps)
            self.previous_timesteps = replayed_timesteps(
                episode,
                frame_stride=self.frame_stride,
                frame_limit=self.frame_limit,
            )
            count = replay_episode(
                self.rr,
                episode,
                episode_index=0,
                namespace="",
                cameras=self.cameras,
                frame_stride=self.frame_stride,
                frame_limit=self.frame_limit,
                topreward_results=self.topreward_results,
            )
        except Exception as exc:
            logger.exception("Failed to load %s", episode)
            self.status.set(f"Failed to load {episode.name}: {exc}")
        else:
            self.status.set(
                f"Loaded {count} timestep(s) from {episode.name}. Use Left/Right arrows, buttons, r=reload, q=quit."
            )
            print(f"Loaded {count} timestep(s) from {episode}")
        finally:
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.previous_button.configure(state=state)
        self.reload_button.configure(state=state)
        self.next_button.configure(state=state)
        if not busy:
            self._update_buttons()

    def _update_buttons(self) -> None:
        self.previous_button.configure(state="normal" if self.index > 0 else "disabled")
        self.next_button.configure(state="normal" if self.index < len(self.episode_dirs) - 1 else "disabled")
        self.reload_button.configure(state="normal")


def _log_episode_metadata(rr: Any, metadata: dict[str, Any], *, namespace: str, episode: Path) -> None:
    if not metadata:
        return
    text = json.dumps(metadata, indent=2, sort_keys=True, default=str)
    if hasattr(rr, "TextDocument"):
        rr.log(f"{namespace}episode_metadata", rr.TextDocument(text, media_type="application/json"))
    elif hasattr(rr, "TextLog"):
        rr.log(f"{namespace}episode_metadata", rr.TextLog(text))
    print(f"Episode {episode.name}: {metadata.get('task_description', '')}")


def _log_vector(rr: Any, base: str, names: Any, values: Any) -> None:
    if values is None:
        return
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    names_list = _as_name_list(names, len(vector))
    for name, value in zip(names_list, vector, strict=False):
        if not math.isfinite(float(value)):
            continue
        rr.log(f"{base}/{rerun_entity_name(name)}", _rerun_scalar(rr, float(value)))


def _log_topreward_scores(rr: Any, *, namespace: str, row: pd.Series) -> None:
    paths = {
        "topreward_logp_true": "interpolated/logp_true",
        "topreward_logp_false": "interpolated/logp_false",
        "topreward_true_false_margin": "interpolated/true_false_margin",
        "topreward_true_normalized_plot": "within_episode_normalized/logp_true",
        "topreward_margin_normalized_plot": "within_episode_normalized/true_false_margin",
    }
    for column, path in paths.items():
        value = _float_or_none(row.get(column))
        if value is not None:
            rr.log(f"{namespace}{path}", _rerun_scalar(rr, value))

    if bool(row.get("is_anchor", False)):
        for column, name in (
            ("topreward_logp_true", "logp_true"),
            ("topreward_logp_false", "logp_false"),
            ("topreward_true_false_margin", "true_false_margin"),
        ):
            value = _float_or_none(row.get(column))
            if value is not None:
                rr.log(f"{namespace}measured_anchors/{name}", _rerun_scalar(rr, value))


def send_topreward_blueprint(rr: Any, *, camera_names: list[str]) -> None:
    """Arrange lazy replay cameras beside raw and normalized reward plots."""

    if not hasattr(rr, "send_blueprint"):
        return
    try:
        rrb = importlib.import_module("rerun.blueprint")
    except ImportError:
        return
    camera_views = [
        rrb.Spatial2DView(origin=f"cameras/{rerun_entity_name(camera)}", name=camera)
        for camera in camera_names
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(*camera_views, name="Cameras"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="topreward",
                    contents=["topreward/interpolated/**", "topreward/measured_anchors/**"],
                    name="TOPReward raw and measured anchors",
                ),
                rrb.TimeSeriesView(
                    origin="topreward/within_episode_normalized",
                    name="TOPReward normalized (plot only)",
                ),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)


def _rerun_scalar(rr: Any, value: float) -> Any:
    if hasattr(rr, "Scalar"):
        return rr.Scalar(value)
    return rr.Scalars(value)


def _as_name_list(names: Any, length: int) -> list[str]:
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (list, tuple)) and len(names) == length:
        return [str(name) for name in names]
    return [f"joint_{index}" for index in range(length)]


def _first_float(df: pd.DataFrame, column: str) -> float:
    if column not in df or len(df) == 0:
        return 0.0
    value = _float_or_none(df[column].iloc[0])
    return 0.0 if value is None else value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Intermediate dataset root containing episode-* directories.")
    parser.add_argument("--episode", action="append", help="Episode name or number to replay. Can be passed more than once.")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to replay after filtering.")
    parser.add_argument("--camera", action="append", help="Only replay this camera name. Can be passed more than once.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Replay every Nth dataset timestep.")
    parser.add_argument("--frame-limit", type=int, help="Maximum timesteps to replay per episode after stride filtering.")
    parser.add_argument(
        "--topreward-dir",
        type=Path,
        help="TOPReward output directory containing timestep_scores.parquet and episode_scores.parquet.",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Open a companion Previous/Next window and load only one episode into Rerun at a time.",
    )
    parser.add_argument("--application-id", default="orbit_intermediate_dataset")
    parser.add_argument("--no-spawn", action="store_true", help="Do not open a local Rerun viewer.")
    parser.add_argument(
        "--rerun-connect-grpc",
        help="Stream to an existing Rerun viewer, e.g. rerun+http://127.0.0.1:9876/proxy.",
    )
    parser.add_argument("--rerun-save", type=Path, help="Write a replayable .rrd recording.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None, rr_module: Any | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be >= 1")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.frame_limit is not None and args.frame_limit < 1:
        parser.error("--frame-limit must be >= 1")

    episode_dirs = discover_episode_dirs(
        args.dataset,
        episodes=args.episode,
        max_episodes=args.max_episodes,
    )
    if not episode_dirs:
        parser.error(f"No intermediate episodes with timesteps.parquet found in: {args.dataset}")

    topreward_results = None
    if args.topreward_dir is not None:
        try:
            topreward_results = load_topreward_results(
                args.topreward_dir,
                dataset_name=args.dataset.expanduser().name,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    if rr_module is None:
        try:
            import rerun as rr_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise SystemExit("Missing dependency: install rerun-sdk to use bimanual-dataset-rerun") from exc

    rr = rr_module
    rr.init(args.application_id, spawn=not args.no_spawn)
    if args.rerun_connect_grpc is not None:
        rr.connect_grpc(args.rerun_connect_grpc)
    if args.rerun_save is not None:
        save_path = args.rerun_save.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path)
    if args.lazy and topreward_results is not None:
        first_df = pd.read_parquet(episode_dirs[0] / "timesteps.parquet")
        blueprint_cameras = camera_names_from_episode(
            episode_dirs[0],
            first_df,
            requested=set(args.camera) if args.camera else None,
        )
        send_topreward_blueprint(rr, camera_names=blueprint_cameras)

    cameras = set(args.camera) if args.camera else None
    if args.lazy:
        navigator = LazyEpisodeNavigator(
            rr,
            episode_dirs,
            application_id=args.application_id,
            rerun_connect_grpc=args.rerun_connect_grpc,
            rerun_save=args.rerun_save,
            reconnect_to_spawned_viewer=not args.no_spawn and args.rerun_connect_grpc is None,
            cameras=cameras,
            frame_stride=int(args.frame_stride),
            frame_limit=args.frame_limit,
            topreward_results=topreward_results,
        )
        navigator.run()
        return

    total_frames = 0
    multiple = len(episode_dirs) > 1
    for episode_index, episode in enumerate(episode_dirs):
        namespace = f"episodes/{rerun_entity_name(episode.name)}/" if multiple else ""
        count = replay_episode(
            rr,
            episode,
            episode_index=episode_index,
            namespace=namespace,
            cameras=cameras,
            frame_stride=int(args.frame_stride),
            frame_limit=args.frame_limit,
            topreward_results=topreward_results,
        )
        total_frames += count
        print(f"Logged {count} timestep(s) from {episode}")
    print(f"Logged {total_frames} total timestep(s) from {len(episode_dirs)} episode(s)")


if __name__ == "__main__":
    main()
