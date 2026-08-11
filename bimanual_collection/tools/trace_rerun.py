"""Replay inference debug trace images into Rerun."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceVideoFrame:
    sequence: int
    timestamp_s: float
    images_by_camera: dict[str, Path]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def rerun_entity_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def discover_trace_dirs(path: Path, *, recursive: bool = False) -> list[Path]:
    """Return trace run directories under path, or path itself if it is one."""

    path = path.expanduser()
    if (path / "events.jsonl").exists():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Trace path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Trace path is not a directory: {path}")

    candidates = path.rglob("events.jsonl") if recursive else path.glob("*/events.jsonl")
    return sorted(event_path.parent for event_path in candidates)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def replay_trace_images(
    rr: Any,
    trace_dir: Path,
    *,
    run_index: int,
    namespace: str,
    cameras: set[str] | None = None,
    image_limit: int | None = None,
) -> int:
    events = iter_jsonl(trace_dir / "events.jsonl")
    logged = 0
    for event in events:
        if event.get("event") != "policy_observation":
            continue
        sequence = int(event["observation_sequence"])
        timestamp_s = float(
            event.get("observation_timestamp_s", event.get("monotonic_timestamp_s", 0.0))
        )
        rr.set_time_sequence("trace_run", run_index)
        rr.set_time_sequence("observation_sequence", sequence)
        rr.set_time_seconds("trace_time", timestamp_s)
        if event.get("wall_timestamp_s") is not None:
            rr.set_time_seconds("wall_time", float(event["wall_timestamp_s"]))

        for camera in event.get("cameras", []):
            camera_name = str(camera.get("camera_name", "camera"))
            if cameras is not None and camera_name not in cameras:
                continue
            image_path = camera.get("image_path")
            if image_path is None:
                continue
            absolute_image_path = trace_dir / image_path
            if not absolute_image_path.exists():
                logger.warning("Missing trace image: %s", absolute_image_path)
                continue
            image = np.asarray(Image.open(absolute_image_path).convert("RGB"))
            entity = f"{namespace}cameras/{rerun_entity_name(camera_name)}/image"
            rr.log(entity, rr.Image(image))
            logged += 1
            if image_limit is not None and logged >= image_limit:
                return logged
    return logged


def collect_trace_video_frames(
    trace_dir: Path,
    *,
    cameras: set[str] | None = None,
    frame_limit: int | None = None,
) -> tuple[list[TraceVideoFrame], list[str]]:
    events = iter_jsonl(trace_dir / "events.jsonl")
    frames: list[TraceVideoFrame] = []
    camera_order: list[str] = []
    for event in events:
        if event.get("event") != "policy_observation":
            continue
        images_by_camera: dict[str, Path] = {}
        for camera in event.get("cameras", []):
            camera_name = str(camera.get("camera_name", "camera"))
            if cameras is not None and camera_name not in cameras:
                continue
            image_path = camera.get("image_path")
            if image_path is None:
                continue
            absolute_image_path = trace_dir / image_path
            if not absolute_image_path.exists():
                logger.warning("Missing trace image: %s", absolute_image_path)
                continue
            images_by_camera[camera_name] = absolute_image_path
            if camera_name not in camera_order:
                camera_order.append(camera_name)
        if not images_by_camera:
            continue
        frames.append(
            TraceVideoFrame(
                sequence=int(event["observation_sequence"]),
                timestamp_s=float(
                    event.get("observation_timestamp_s", event.get("monotonic_timestamp_s", 0.0))
                ),
                images_by_camera=images_by_camera,
            )
        )
        if frame_limit is not None and len(frames) >= frame_limit:
            break
    return frames, camera_order


def make_tiled_video_frame(
    frame: TraceVideoFrame,
    camera_order: list[str],
    *,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    from PIL import ImageDraw

    label_height = 24
    cols = max(1, math.ceil(math.sqrt(len(camera_order))))
    rows = max(1, math.ceil(len(camera_order) / cols))
    canvas = Image.new("RGB", (cols * tile_width, rows * (tile_height + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, camera_name in enumerate(camera_order):
        x = (index % cols) * tile_width
        y = (index // cols) * (tile_height + label_height)
        image_path = frame.images_by_camera.get(camera_name)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((tile_width, tile_height))
            paste_x = x + (tile_width - image.width) // 2
            paste_y = y + (tile_height - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
        else:
            draw.text((x + 8, y + 8), "missing", fill="white")
        draw.rectangle(
            (x, y + tile_height, x + tile_width, y + tile_height + label_height),
            fill="black",
        )
        draw.text(
            (x + 8, y + tile_height + 5),
            f"{camera_name}  seq={frame.sequence}",
            fill="white",
        )
    return np.asarray(canvas)


def export_trace_video(
    trace_dir: Path,
    output_path: Path,
    *,
    cameras: set[str] | None = None,
    fps: float = 10.0,
    frame_limit: int | None = None,
    tile_width: int = 320,
    tile_height: int = 240,
) -> int:
    import cv2  # type: ignore

    frames, camera_order = collect_trace_video_frames(trace_dir, cameras=cameras, frame_limit=frame_limit)
    if not frames:
        raise ValueError(f"No trace images found in {trace_dir}")

    first_frame = make_tiled_video_frame(frames[0], camera_order, tile_width=tile_width, tile_height=tile_height)
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        for frame in frames:
            rgb = make_tiled_video_frame(frame, camera_order, tile_width=tile_width, tile_height=tile_height)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return len(frames)


def video_output_path(video_out: Path, trace_dir: Path, *, multiple_runs: bool) -> Path:
    video_out = video_out.expanduser()
    if multiple_runs or video_out.suffix.lower() != ".mp4":
        return video_out / f"{trace_dir.name}.mp4"
    return video_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Single trace run directory, or a parent directory containing run-* trace dirs.",
    )
    parser.add_argument("--recursive", action="store_true", help="Find trace runs recursively under --trace-dir.")
    parser.add_argument(
        "--camera",
        action="append",
        help="Only replay this camera name. Can be passed more than once.",
    )
    parser.add_argument("--image-limit", type=int, help="Maximum number of images to log per run.")
    parser.add_argument(
        "--video-out",
        type=Path,
        help="Write MP4 video directly from trace images. For multiple runs, this must be an output directory.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for --video-out MP4 export.")
    parser.add_argument("--video-frame-limit", type=int, help="Maximum number of video frames to write per run.")
    parser.add_argument(
        "--video-tile-width",
        type=int,
        default=320,
        help="Width of each camera tile in exported video.",
    )
    parser.add_argument(
        "--video-tile-height",
        type=int,
        default=240,
        help="Height of each camera tile in exported video.",
    )
    parser.add_argument("--application-id", default="orbit_inference_trace_images")
    parser.add_argument("--no-rerun", action="store_true", help="Skip Rerun replay and only write --video-out.")
    parser.add_argument("--no-spawn", action="store_true", help="Do not open a local Rerun viewer.")
    parser.add_argument(
        "--rerun-connect-grpc",
        help="Stream to an existing Rerun viewer, e.g. rerun+http://127.0.0.1:9876/proxy.",
    )
    parser.add_argument("--rerun-save", type=Path, help="Write a replayable .rrd recording.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.image_limit is not None and args.image_limit < 1:
        parser.error("--image-limit must be >= 1")
    if args.video_frame_limit is not None and args.video_frame_limit < 1:
        parser.error("--video-frame-limit must be >= 1")
    if args.video_fps <= 0:
        parser.error("--video-fps must be > 0")
    if args.video_tile_width < 1 or args.video_tile_height < 1:
        parser.error("--video-tile-width and --video-tile-height must be >= 1")
    if args.no_rerun and args.video_out is None:
        parser.error("--no-rerun requires --video-out")
    if args.no_rerun and (args.rerun_connect_grpc is not None or args.rerun_save is not None):
        parser.error("--no-rerun cannot be combined with Rerun output options")

    trace_dirs = discover_trace_dirs(args.trace_dir, recursive=bool(args.recursive))
    if not trace_dirs:
        parser.error(f"No trace runs with events.jsonl found under {args.trace_dir}")
    if (
        args.video_out is not None
        and len(trace_dirs) > 1
        and args.video_out.suffix.lower() == ".mp4"
    ):
        parser.error("--video-out must be a directory when exporting multiple trace runs")

    cameras = set(args.camera) if args.camera else None
    if args.video_out is not None:
        total_frames = 0
        for trace_dir in trace_dirs:
            output_path = video_output_path(args.video_out, trace_dir, multiple_runs=len(trace_dirs) > 1)
            count = export_trace_video(
                trace_dir,
                output_path,
                cameras=cameras,
                fps=float(args.video_fps),
                frame_limit=args.video_frame_limit,
                tile_width=int(args.video_tile_width),
                tile_height=int(args.video_tile_height),
            )
            total_frames += count
            print(f"Wrote {count} video frames to {output_path}")
        print(f"Wrote {total_frames} total video frames from {len(trace_dirs)} trace run(s)")

    if args.no_rerun:
        return

    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: install rerun-sdk to use bimanual-trace-rerun") from exc

    rr.init(args.application_id, spawn=not args.no_spawn)
    if args.rerun_connect_grpc is not None:
        rr.connect_grpc(args.rerun_connect_grpc)
    if args.rerun_save is not None:
        save_path = args.rerun_save.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path)

    total_images = 0
    for run_index, trace_dir in enumerate(trace_dirs):
        namespace = f"runs/{rerun_entity_name(trace_dir.name)}/" if len(trace_dirs) > 1 else ""
        count = replay_trace_images(
            rr,
            trace_dir,
            run_index=run_index,
            namespace=namespace,
            cameras=cameras,
            image_limit=args.image_limit,
        )
        total_images += count
        print(f"Logged {count} images from {trace_dir}")
    print(f"Logged {total_images} total images from {len(trace_dirs)} trace run(s)")


if __name__ == "__main__":
    main()
