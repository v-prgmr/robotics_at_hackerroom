"""Export Orbit intermediate datasets to LeRobot dataset format."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def intermediate_episodes(root: Path) -> list[Path]:
    """Return published intermediate episode directories."""

    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "timesteps.parquet").exists()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Intermediate dataset root containing episode-* directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination LeRobot dataset root. Must not exist unless --overwrite is passed.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot repo id to store in metadata, for example 'vrazer/teabags_kitting_50_v1'.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Robot/control FPS for the LeRobot dataset.")
    parser.add_argument(
        "--video-codec",
        default="h264",
        choices=["auto", "h264", "h264_nvenc", "h264_qsv", "h264_vaapi", "h264_videotoolbox", "hevc", "hevc_nvenc", "hevc_videotoolbox", "libsvtav1"],
        help="LeRobot video encoder. Default h264 uses much less memory than libsvtav1.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=1,
        help="Encoder threads passed to LeRobot. Use 1 for lowest memory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output-dir before exporting if it already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    exporter: Callable[..., Any] | None = None,
) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.encoder_threads < 1:
        parser.error("--encoder-threads must be >= 1")
    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        parser.error(f"Input path is not a directory: {input_dir}")
    episodes = intermediate_episodes(input_dir)
    if not episodes:
        parser.error(f"No intermediate episodes with timesteps.parquet found in: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        logger.info("Removing existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)

    if exporter is None:
        from bimanual_collection.recording.backends.lerobot_export import export_to_lerobot

        exporter = export_to_lerobot

    print(f"Exporting {len(episodes)} episode(s) from {input_dir}")
    print(f"LeRobot output: {output_dir}")
    print(f"Repo id: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"Video codec: {args.video_codec}")
    print(f"Encoder threads: {args.encoder_threads}")
    exporter(
        input_dir,
        output_dir,
        args.repo_id,
        int(args.fps),
        vcodec=args.video_codec,
        encoder_threads=int(args.encoder_threads),
    )
    print(f"Export complete: {output_dir}")
    print("View with:")
    print(f"  uv run lerobot-dataset-viz --repo-id {args.repo_id} --root {output_dir} --episode-index 0")


if __name__ == "__main__":
    main()
