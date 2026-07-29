"""Install Orbit OpenPI adapters, compute norm stats, and launch pi0.5 training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

try:
    from openpi_orbit.export_lerobot_v21 import export_openpi_lerobot_v21
    from openpi_orbit.install_into_openpi import install
except ImportError:  # Direct `python openpi_orbit/train_pi05_orbit.py` execution.
    from export_lerobot_v21 import export_openpi_lerobot_v21
    from install_into_openpi import install


def _default_lerobot_home() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))


def _link_dataset(dataset_root: Path, repo_id: str, *, replace: bool) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    target = _default_lerobot_home() / repo_id
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == dataset_root:
                return target
        except FileNotFoundError:
            pass
        if target.is_symlink():
            print(f"Replacing stale LeRobot cache symlink: {target} -> {dataset_root}")
            target.unlink()
        elif not replace:
            raise FileExistsError(
                f"LeRobot cache target already exists: {target}. "
                "Use --replace-dataset-link to replace an existing symlink."
            )
        else:
            raise FileExistsError(f"Refusing to replace non-symlink dataset cache path: {target}")

    target.symlink_to(dataset_root, target_is_directory=True)
    return target


def _is_openpi_compatible_lerobot(root: Path) -> bool:
    return (
        (root / "meta/info.json").exists()
        and (root / "meta/tasks.jsonl").exists()
        and (root / "meta/episodes.jsonl").exists()
    )


def _is_lerobot_v3(root: Path) -> bool:
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return False
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file).get("codebase_version") == "v3.0"


def _prepare_dataset_root(dataset_root: Path, repo_id: str, *, overwrite_export: bool) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if _is_openpi_compatible_lerobot(dataset_root):
        return dataset_root
    if _is_lerobot_v3(dataset_root):
        output_root = dataset_root.with_name(f"{dataset_root.name}_openpi_v21")
        if output_root.exists() and _is_openpi_compatible_lerobot(output_root) and not overwrite_export:
            print(f"Using existing OpenPI-compatible export: {output_root}")
            return output_root
        print(f"Converting LeRobot v3 export for OpenPI: {dataset_root} -> {output_root}")
        return export_openpi_lerobot_v21(dataset_root, output_root, repo_id=repo_id, overwrite=True)
    raise ValueError(
        f"Dataset root is not an OpenPI-compatible LeRobot dataset or Orbit LeRobot v3 export: {dataset_root}"
    )


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-dir", type=Path, required=True, help="Path to the OpenPI checkout.")
    parser.add_argument("--dataset-root", type=Path, help="Local Orbit LeRobot export root to symlink into the cache.")
    parser.add_argument("--repo-id", default="local/orbit_so100", help="LeRobot repo id used by the OpenPI config.")
    parser.add_argument("--config-name", default="pi05_orbit_so100", help="OpenPI train config name.")
    parser.add_argument("--exp-name", default="orbit_pi05", help="OpenPI experiment/checkpoint name.")
    parser.add_argument("--steps", type=int, default=20_000, help="Training steps passed to OpenPI.")
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size passed to OpenPI.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers passed to OpenPI.")
    parser.add_argument("--save-interval", type=int, default=2_000, help="Checkpoint save interval.")
    parser.add_argument("--log-interval", type=int, default=100, help="Training log interval.")
    parser.add_argument("--max-norm-frames", type=int, help="Optional frame cap for norm-stat computation.")
    parser.add_argument("--xla-mem-fraction", default="0.9", help="XLA_PYTHON_CLIENT_MEM_FRACTION value.")
    parser.add_argument("--skip-norm-stats", action="store_true", help="Skip compute_norm_stats.py.")
    parser.add_argument(
        "--overwrite-openpi-export",
        action="store_true",
        help="Regenerate the OpenPI-compatible v2.1 export when --dataset-root points at a v3 export.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to OpenPI training.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to OpenPI training.")
    parser.add_argument(
        "--replace-dataset-link",
        action="store_true",
        help="Replace an existing LeRobot cache symlink for --repo-id.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    openpi_dir = args.openpi_dir.expanduser().resolve()

    install(openpi_dir)

    if args.dataset_root is not None:
        dataset_root = _prepare_dataset_root(
            args.dataset_root,
            args.repo_id,
            overwrite_export=args.overwrite_openpi_export,
        )
        target = _link_dataset(dataset_root, args.repo_id, replace=args.replace_dataset_link)
        print(f"LeRobot cache link: {target} -> {target.resolve()}")

    env = os.environ.copy()
    env["ORBIT_LEROBOT_REPO_ID"] = args.repo_id
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction

    if not args.skip_norm_stats:
        norm_command = ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", args.config_name]
        if args.max_norm_frames is not None:
            norm_command.extend(["--max-frames", str(args.max_norm_frames)])
        _run(norm_command, cwd=openpi_dir, env=env)

    train_command = [
        "uv",
        "run",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--num-train-steps={args.steps}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--save-interval={args.save_interval}",
        f"--log-interval={args.log_interval}",
    ]
    if args.overwrite:
        train_command.append("--overwrite")
    if args.resume:
        train_command.append("--resume")

    _run(train_command, cwd=openpi_dir, env=env)


if __name__ == "__main__":
    main()
