"""Install Orbit ManiFlow bridge files into a ManiFlow checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Target exists: {target}. Pass --overwrite to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Installed {target}")


def install(maniflow_dir: Path, *, overwrite: bool = False) -> None:
    source_dir = Path(__file__).resolve().parent
    maniflow_dir = maniflow_dir.expanduser().resolve()
    package_dir = maniflow_dir / "maniflow"
    if not package_dir.exists():
        raise FileNotFoundError(f"Expected ManiFlow package directory at {package_dir}")

    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Installed {init_file}")

    files = [
        (
            source_dir / "maniflow_dataset/orbit_image_dataset.py",
            package_dir / "dataset/orbit_image_dataset.py",
        ),
        (
            source_dir / "maniflow_config/maniflow_image_orbit.yaml",
            package_dir / "config/maniflow_image_orbit.yaml",
        ),
        (
            source_dir / "maniflow_config/robotwin_task/orbit_so100_image.yaml",
            package_dir / "config/robotwin_task/orbit_so100_image.yaml",
        ),
        (
            source_dir / "maniflow_workspace/train_maniflow_orbit_workspace.py",
            package_dir / "workspace/train_maniflow_orbit_workspace.py",
        ),
    ]
    for source, target in files:
        _copy_file(source, target, overwrite=overwrite)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maniflow-dir", type=Path, required=True, help="Path to allenai/maniflow checkout.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing installed bridge files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    install(args.maniflow_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
