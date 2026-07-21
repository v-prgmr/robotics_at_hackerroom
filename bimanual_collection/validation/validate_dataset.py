"""Validate intermediate bimanual datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ValidationReport:
    """Aggregates validation errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_episode(path: Path, max_jitter_s: float, max_stale_fraction: float) -> ValidationReport:
    report = ValidationReport()
    metadata_path = path / "episode_metadata.json"
    timesteps_path = path / "timesteps.parquet"
    camera_index_path = path / "camera_index.parquet"
    if not metadata_path.exists():
        report.add_error(f"Missing episode_metadata.json in {path}")
    if not timesteps_path.exists():
        report.add_error(f"Missing timesteps.parquet in {path}")
        return report
    if not camera_index_path.exists():
        report.add_error(f"Missing camera_index.parquet in {path}")

    df = pd.read_parquet(timesteps_path)
    if df.empty:
        report.add_error(f"Episode has no timesteps: {path}")
        return report
    if not df["monotonic_timestamp_s"].is_monotonic_increasing:
        report.add_error(f"Non-monotonic timestamps: {path}")
    missing_robot_values = df[
        ["left_follower_joints", "right_follower_joints", "left_commanded_action", "right_commanded_action"]
    ].isna()
    if bool(missing_robot_values.to_numpy().any()):
        report.add_error(f"Missing robot states or actions: {path}")

    left_dim = df["left_follower_joints"].map(len).nunique()
    right_dim = df["right_follower_joints"].map(len).nunique()
    left_action_dim = df["left_commanded_action"].map(len).nunique()
    right_action_dim = df["right_commanded_action"].map(len).nunique()
    if any(value != 1 for value in (left_dim, right_dim, left_action_dim, right_action_dim)):
        report.add_error(f"Inconsistent left/right observation or action dimensions: {path}")
    if df["left_follower_joints"].map(len).iloc[0] != df["right_follower_joints"].map(len).iloc[0]:
        report.add_error(f"Left/right observation dimensions differ: {path}")
    if df["left_commanded_action"].map(len).iloc[0] != df["right_commanded_action"].map(len).iloc[0]:
        report.add_error(f"Left/right action dimensions differ: {path}")

    intervals = df["monotonic_timestamp_s"].diff().dropna()
    if not intervals.empty and (intervals - intervals.median()).abs().max() > max_jitter_s:
        report.add_warning(f"Control-loop jitter exceeds {max_jitter_s:.4f}s: {path}")

    if camera_index_path.exists():
        cdf = pd.read_parquet(camera_index_path)
        if bool(cdf["missing"].to_numpy().any()):
            report.add_error(f"Missing camera frames in {path}")
        stale_fraction = float(cdf["stale"].mean()) if len(cdf) else 0.0
        if stale_fraction > max_stale_fraction:
            report.add_error(f"Stale camera frame fraction {stale_fraction:.3f} exceeds {max_stale_fraction:.3f}: {path}")
        for video_path in cdf["video_path"].dropna().unique():
            full = path / video_path
            if not full.exists():
                report.add_error(f"Missing video file referenced by index: {full}")

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if int(metadata.get("num_timesteps", -1)) != len(df):
            report.add_error(f"Episode length mismatch in metadata: {path}")
    return report


def validate_dataset(root: Path, max_jitter_s: float = 0.02, max_stale_fraction: float = 0.05) -> ValidationReport:
    report = ValidationReport()
    episodes = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not episodes:
        report.add_error(f"No episode directories found in {root}")
        return report
    for episode in episodes:
        ep_report = validate_episode(episode, max_jitter_s=max_jitter_s, max_stale_fraction=max_stale_fraction)
        report.errors.extend(ep_report.errors)
        report.warnings.extend(ep_report.warnings)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-jitter-s", type=float, default=0.02)
    parser.add_argument("--max-stale-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset, args.max_jitter_s, args.max_stale_fraction)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.ok:
        print("Dataset validation passed")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
