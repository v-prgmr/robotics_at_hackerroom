"""Interactive calibration utility for bimanual SO-100 leaders and followers."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

ArmRole = Literal["left_follower", "right_follower", "left_leader", "right_leader"]


@dataclass(frozen=True)
class CalibrationRole:
    """Mapping from bimanual role to LeRobot device constructor inputs."""

    role: ArmRole
    config_port_key: str
    default_id: str
    device_kind: Literal["follower", "leader"]


ROLES: dict[ArmRole, CalibrationRole] = {
    "left_follower": CalibrationRole("left_follower", "left_robot_port", "left_follower", "follower"),
    "right_follower": CalibrationRole("right_follower", "right_robot_port", "right_follower", "follower"),
    "left_leader": CalibrationRole("left_leader", "left_leader_port", "left_leader", "leader"),
    "right_leader": CalibrationRole("right_leader", "right_leader_port", "right_leader", "leader"),
}


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def cfg_value(args: argparse.Namespace, config: dict, name: str, default=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name.replace("_", "-"), config.get(name, default))


def role_id(role: CalibrationRole, config: dict) -> str:
    ids = config.get("calibration_ids", {}) or {}
    if not isinstance(ids, dict):
        raise ValueError("calibration_ids must be a mapping")
    return str(ids.get(role.role, role.default_id))


def make_device(role: CalibrationRole, port: str, device_id: str, calibration_dir: Path | None, use_degrees: bool):
    """Construct one LeRobot SO-100 arm for calibration."""

    if role.device_kind == "follower":
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        return SO100Follower(
            SO100FollowerConfig(
                id=device_id,
                port=port,
                calibration_dir=calibration_dir,
                cameras={},
                use_degrees=use_degrees,
                disable_torque_on_disconnect=True,
            )
        )

    from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

    return SO100Leader(
        SO100LeaderConfig(
            id=device_id,
            port=port,
            calibration_dir=calibration_dir,
            use_degrees=use_degrees,
        )
    )


def default_home_positions_path(config: dict, args: argparse.Namespace) -> Path:
    raw = cfg_value(args, config, "home_position_file")
    if raw is None:
        control = config.get("teleop_control", {}) or {}
        if not isinstance(control, dict):
            raise ValueError("teleop_control must be a mapping")
        raw = control.get("home_position_file")
    if raw:
        return Path(raw).expanduser()
    calibration_dir_raw = cfg_value(args, config, "calibration_dir")
    calibration_dir = Path(calibration_dir_raw).expanduser() if calibration_dir_raw else Path("calibration/so100_bimanual")
    return calibration_dir / "home_positions.yaml"


def read_device_pose(device, role: CalibrationRole) -> dict[str, float]:
    """Read a calibrated arm pose using the appropriate LeRobot API."""

    if role.device_kind == "follower":
        pose = device.get_observation()
    else:
        pose = device.get_action()
    return {key: float(value) for key, value in pose.items() if key.endswith(".pos")}


def save_home_pose(path: Path, role: CalibrationRole, pose: dict[str, float], device_id: str, port: str) -> None:
    """Upsert one role's safe home pose."""

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    else:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Home pose file must contain a mapping: {path}")
    data.setdefault("version", 1)
    data.setdefault("home_positions", {})
    data["home_positions"][role.role] = {
        "device_kind": role.device_kind,
        "id": device_id,
        "port": port,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "joints": pose,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def calibrate_role(role: CalibrationRole, config: dict, args: argparse.Namespace) -> tuple[Path, dict[str, float] | None]:
    """Run LeRobot's interactive calibration flow for one arm role."""

    port = cfg_value(args, config, role.config_port_key)
    if not port:
        raise ValueError(f"Missing `{role.config_port_key}` for {role.role}")

    calibration_dir_raw = cfg_value(args, config, "calibration_dir")
    calibration_dir = Path(calibration_dir_raw).expanduser() if calibration_dir_raw else None
    device_id = role_id(role, config)
    use_degrees = bool(cfg_value(args, config, "use_degrees", True))

    device = make_device(role, str(port), device_id, calibration_dir, use_degrees)
    print(f"\n=== Calibrating {role.role.replace('_', ' ')} ===")
    print(f"Port: {port}")
    print(f"Calibration id: {device_id}")
    print(f"Calibration file: {device.calibration_fpath}")
    input("Connect only the requested arm if possible, clear the workspace, then press Enter...")

    device.connect(calibrate=False)
    home_pose = None
    try:
        device.calibrate()
        if not args.skip_home_capture:
            print(f"\nMove {role.role.replace('_', ' ')} to its SAFE HOME pose.")
            print("For followers, choose a collision-free start pose. For leaders, match the corresponding follower home pose.")
            input("Press Enter to capture this home pose...")
            home_pose = read_device_pose(device, role)
            home_path = default_home_positions_path(config, args)
            save_home_pose(home_path, role, home_pose, device_id, str(port))
            print(f"Saved home pose for {role.role} to {home_path}")
    finally:
        device.disconnect()
    return device.calibration_fpath, home_pose


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config containing arm ports.")
    parser.add_argument("--calibration-dir", type=Path, help="Directory for LeRobot calibration JSON files.")
    parser.add_argument("--home-position-file", type=Path, help="YAML file for safe home poses.")
    parser.add_argument("--skip-home-capture", action="store_true", help="Only run LeRobot calibration; do not capture home poses.")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(ROLES),
        default=list(ROLES),
        help="Roles to calibrate in order.",
    )
    parser.add_argument("--left-robot-port")
    parser.add_argument("--right-robot-port")
    parser.add_argument("--left-leader-port")
    parser.add_argument("--right-leader-port")
    parser.add_argument("--use-degrees", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    config = load_config(args.config)

    saved: dict[str, str] = {}
    homes: dict[str, dict[str, float]] = {}
    for role_name in args.roles:
        role = ROLES[role_name]
        calibration_path, home_pose = calibrate_role(role, config, args)
        saved[role.role] = str(calibration_path)
        if home_pose is not None:
            homes[role.role] = home_pose

    print("\nCalibration files:")
    for role, path in saved.items():
        print(f"  {role}: {path}")
    if homes:
        print(f"\nHome poses: {default_home_positions_path(config, args)}")


if __name__ == "__main__":
    main()
