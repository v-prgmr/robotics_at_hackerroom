"""Backfill task descriptions into recorded Orbit intermediate episodes."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_episode_selector(values: list[str] | None) -> set[str] | None:
    """Parse episode selectors like '1', '000003', 'episode-000004', or '7-10'."""

    if not values:
        return None
    selected: set[str] = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        if "-" in token and not token.startswith("episode-"):
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"Invalid descending episode range: {token}")
            selected.update(f"episode-{index:06d}" for index in range(start, end + 1))
            continue
        if token.startswith("episode-"):
            selected.add(token)
        else:
            selected.add(f"episode-{int(token):06d}")
    return selected


def find_episode_metadata(dataset_dir: Path, selected: set[str] | None = None) -> list[Path]:
    """Find published intermediate episode metadata files."""

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")
    metadata_paths = []
    for episode_dir in sorted(dataset_dir.iterdir()):
        if not episode_dir.is_dir() or episode_dir.name.startswith("."):
            continue
        if selected is not None and episode_dir.name not in selected:
            continue
        metadata_path = episode_dir / "episode_metadata.json"
        if metadata_path.exists():
            metadata_paths.append(metadata_path)
    return metadata_paths


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via same-directory temp file then atomic replace."""

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        tmp_path = Path(file.name)
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    tmp_path.replace(path)


def set_task_description(
    dataset_dir: Path,
    task: str,
    *,
    selected: set[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Update task_description in selected intermediate episode metadata files."""

    if not task.strip():
        raise ValueError("Task description must not be empty")
    metadata_paths = find_episode_metadata(dataset_dir, selected=selected)
    if selected is not None:
        found = {path.parent.name for path in metadata_paths}
        missing = sorted(selected - found)
        if missing:
            raise FileNotFoundError(f"Selected episodes were not found or lack metadata: {missing}")
    if not metadata_paths:
        raise ValueError(f"No episode_metadata.json files found in: {dataset_dir}")

    updated: list[Path] = []
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Episode metadata must be a JSON object: {metadata_path}")
        old_task = metadata.get("task_description", "")
        metadata["task_description"] = task
        updated.append(metadata_path)
        print(f"{metadata_path.parent.name}: {old_task!r} -> {task!r}")
        if not dry_run:
            atomic_write_json(metadata_path, metadata)
    return updated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "--dataset-dir",
        dest="dataset_dir",
        type=Path,
        required=True,
        help="Orbit intermediate dataset root containing episode-* directories.",
    )
    parser.add_argument("--task", required=True, help="Task/instruction text to write into each episode.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="Optional episode selectors, e.g. 1 2 7-10 episode-000012. Defaults to all episodes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        selected = parse_episode_selector(args.episodes)
        updated = set_task_description(
            args.dataset_dir.expanduser(),
            args.task,
            selected=selected,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        parser.error(str(exc))
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(updated)} episode(s). Re-export to LeRobot format for training/viewing.")


if __name__ == "__main__":
    main()
