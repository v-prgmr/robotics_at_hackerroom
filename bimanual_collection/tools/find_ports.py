"""Interactive serial-port mapper for bimanual SO-100 setups."""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_ROLE_TO_CONFIG_KEY = {
    "left_follower": "left_robot_port",
    "right_follower": "right_robot_port",
    "left_leader": "left_leader_port",
    "right_leader": "right_leader_port",
}


@dataclass(frozen=True)
class SerialPortInfo:
    """Serial port metadata returned by pyserial."""

    device: str
    description: str
    hwid: str
    stable_path: str | None = None


def list_serial_ports() -> dict[str, SerialPortInfo]:
    """Return available serial ports keyed by device path."""

    from serial.tools import list_ports

    ports: dict[str, SerialPortInfo] = {}
    for port in list_ports.comports():
        ports[port.device] = SerialPortInfo(
            device=port.device,
            description=port.description or "",
            hwid=port.hwid or "",
            stable_path=stable_serial_alias(port.device),
        )
    return ports


def stable_serial_alias(device: str) -> str | None:
    """Return a stable `/dev/serial/by-id` or `/dev/serial/by-path` alias for a tty device."""

    if platform.system() != "Linux":
        return None
    target = Path(device)
    if not target.exists():
        return None
    resolved = target.resolve()
    for base in (Path("/dev/serial/by-id"), Path("/dev/serial/by-path")):
        if not base.exists():
            continue
        for candidate in sorted(base.iterdir()):
            try:
                if candidate.resolve() == resolved:
                    return str(candidate)
            except OSError:
                continue
    return None


def display_ports(ports: dict[str, SerialPortInfo]) -> None:
    if not ports:
        print("No serial ports detected.")
        return
    for info in sorted(ports.values(), key=lambda item: item.device):
        stable = f" stable={info.stable_path}" if info.stable_path else ""
        print(f"  {info.device}{stable}\n    {info.description}\n    {info.hwid}")


def wait_for_enter(message: str) -> None:
    input(f"\n{message}\nPress Enter when ready...")


def identify_role(role: str, prefer_stable: bool, retry_delay_s: float) -> str:
    """Identify one arm role by comparing ports before and after unplugging."""

    while True:
        print(f"\n=== Identify {role.replace('_', ' ')} ===")
        before = list_serial_ports()
        print("Ports before unplugging:")
        display_ports(before)

        wait_for_enter(f"Unplug ONLY the USB cable for {role.replace('_', ' ')}.")
        time.sleep(retry_delay_s)
        after = list_serial_ports()
        removed = sorted(set(before) - set(after))

        if len(removed) == 1:
            device = removed[0]
            info = before[device]
            selected = info.stable_path if prefer_stable and info.stable_path else device
            print(f"Detected {role}: {selected}")
            wait_for_enter(f"Reconnect {role.replace('_', ' ')}.")
            return selected

        print("Could not uniquely identify the port.")
        if not removed:
            print("No port disappeared. Make sure exactly one USB cable was unplugged.")
        else:
            print("More than one port disappeared:")
            for device in removed:
                print(f"  {device}")
        answer = input("Retry this role? [Y/n] ").strip().lower()
        if answer == "n":
            raise RuntimeError(f"Unable to identify {role}")


def write_yaml(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(mapping, file, sort_keys=False)
    print(f"\nWrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("config/local_ports.yaml"))
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(DEFAULT_ROLE_TO_CONFIG_KEY),
        default=list(DEFAULT_ROLE_TO_CONFIG_KEY),
        help="Roles to identify in order.",
    )
    parser.add_argument("--list", action="store_true", help="List serial ports and exit.")
    parser.add_argument("--prefer-raw", action="store_true", help="Write /dev/tty* paths instead of stable aliases.")
    parser.add_argument("--retry-delay-s", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.list:
        display_ports(list_serial_ports())
        return

    mapping: dict[str, str] = {}
    for role in args.roles:
        config_key = DEFAULT_ROLE_TO_CONFIG_KEY[role]
        mapping[config_key] = identify_role(
            role=role,
            prefer_stable=not args.prefer_raw,
            retry_delay_s=args.retry_delay_s,
        )
    write_yaml(args.output, mapping)


if __name__ == "__main__":
    main()
