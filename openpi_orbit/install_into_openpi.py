"""Install Orbit OpenPI adapters into an OpenPI checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ORBIT_IMPORT = "import openpi.training.orbit_config as orbit_config"
ORBIT_CONFIG_SPLICE = "    *orbit_config.get_orbit_configs(),"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_openpi_dir(openpi_dir: Path) -> None:
    expected = [
        openpi_dir / "src/openpi/policies",
        openpi_dir / "src/openpi/training/config.py",
        openpi_dir / "scripts/train.py",
        openpi_dir / "scripts/compute_norm_stats.py",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Not an OpenPI checkout or missing files: " + ", ".join(missing))


def _copy_adapter_files(openpi_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "orbit_policy.py", openpi_dir / "src/openpi/policies/orbit_policy.py")
    shutil.copy2(source_dir / "orbit_data_config.py", openpi_dir / "src/openpi/training/orbit_config.py")


def _patch_config(openpi_dir: Path) -> None:
    config_path = openpi_dir / "src/openpi/training/config.py"
    text = config_path.read_text(encoding="utf-8")

    if ORBIT_IMPORT not in text:
        marker = "_CONFIGS = ["
        if marker not in text:
            raise ValueError(f"Could not find {marker!r} in {config_path}")
        text = text.replace(marker, f"{ORBIT_IMPORT}\n{marker}", 1)

    if ORBIT_CONFIG_SPLICE not in text:
        marker = "_CONFIGS = [\n"
        if marker not in text:
            raise ValueError(f"Could not find {marker!r} in {config_path}")
        text = text.replace(marker, f"{marker}{ORBIT_CONFIG_SPLICE}\n", 1)

    config_path.write_text(text, encoding="utf-8")


def install(openpi_dir: Path) -> None:
    openpi_dir = openpi_dir.expanduser().resolve()
    _require_openpi_dir(openpi_dir)
    _copy_adapter_files(openpi_dir)
    _patch_config(openpi_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", type=Path, required=True, help="Path to the OpenPI checkout.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    install(args.openpi_dir)
    print(f"Installed Orbit OpenPI adapters into {args.openpi_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
