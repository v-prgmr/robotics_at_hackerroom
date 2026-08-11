"""Live Orbit robot deployment using a remote ManiFlow policy server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bimanual_collection.inference import run_inference
from bimanual_collection.bimanual_teleop import cfg_value, load_config
from bimanual_collection.maniflow_remote import (
    DEFAULT_MANIFLOW_CAMERAS,
    ManiFlowRemoteObservationAdapter,
    ManiFlowRemotePolicyRuntime,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--maniflow-server",
        help="ManiFlow policy server URL, e.g. http://127.0.0.1:8765",
    )
    parser.add_argument(
        "--maniflow-timeout-s",
        type=float,
        help="HTTP timeout for policy server requests",
    )
    parser.add_argument(
        "--maniflow-camera",
        action="append",
        dest="maniflow_cameras",
        help="Camera key expected by ManiFlow; defaults to overhead/left_wrist/right_wrist",
    )
    parser.add_argument(
        "--capture-dataset",
        action="store_true",
        help="Capture each armed autonomous rollout until it is labeled success or failure",
    )
    parser.add_argument("--failure-output-dir", type=Path, help="Orbit dataset root for failed rollouts")
    parser.add_argument("--success-output-dir", type=Path, help="Orbit dataset root for successful rollouts")
    parser.add_argument("--failure-key", default="f", help="Classify and save the current rollout as failure")
    parser.add_argument("--success-key", default="s", help="Classify and save the current rollout as success")


def build_runtime_and_adapter(
    args: argparse.Namespace,
    task_description: str,
) -> tuple[ManiFlowRemotePolicyRuntime, ManiFlowRemoteObservationAdapter]:
    config = load_config(args.config)
    server_url = str(cfg_value(args, config, "maniflow_server", "http://127.0.0.1:8765"))
    timeout_s = float(cfg_value(args, config, "maniflow_timeout_s", 10.0))
    configured_cameras = config.get("maniflow_cameras")
    if timeout_s <= 0:
        raise ValueError("maniflow_timeout_s must be > 0")

    cameras = tuple(args.maniflow_cameras or configured_cameras or DEFAULT_MANIFLOW_CAMERAS)
    runtime = ManiFlowRemotePolicyRuntime(
        _normalize_server_url(server_url),
        timeout_s=timeout_s,
        cameras=cameras,
    )
    adapter = ManiFlowRemoteObservationAdapter(
        task_description=task_description,
        cameras=cameras,
    )
    return runtime, adapter


def _normalize_server_url(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"http://{value}"


def main(argv: list[str] | None = None) -> None:
    run_inference(
        argv,
        checkpoint_required=False,
        configure_parser=configure_parser,
        runtime_adapter_builder=build_runtime_and_adapter,
        policy_label="ManiFlow remote",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
