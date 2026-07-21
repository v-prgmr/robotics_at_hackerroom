"""Interactive camera mapper for overhead and wrist cameras."""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


DEFAULT_ROLE_TO_CONFIG_KEY = {
    "overhead": "overhead_camera",
    "left_wrist": "left_wrist_camera",
    "right_wrist": "right_wrist_camera",
}


@dataclass(frozen=True)
class CameraCandidate:
    """Camera device that successfully produced a frame."""

    index: int
    device: str | int
    selected_device: str | int
    resolved_device: str | int
    sample_path: Path
    width: int
    height: int
    fps: float
    backend: str
    stable_path: str | None = None


def stable_camera_alias(device: str | int) -> str | None:
    """Return a stable `/dev/v4l/by-path` or `/dev/v4l/by-id` alias for a camera.

    `by-path` is preferred because identical USB cameras often expose identical
    or missing serial IDs but still have unique physical USB paths.
    """

    if platform.system() != "Linux" or not isinstance(device, str) or not device.startswith("/dev/video"):
        return None
    target = Path(device)
    if not target.exists():
        return None
    resolved = target.resolve()
    for base in (Path("/dev/v4l/by-path"), Path("/dev/v4l/by-id")):
        if not base.exists():
            continue
        for candidate in sorted(base.iterdir()):
            try:
                if candidate.resolve() == resolved:
                    return str(candidate)
            except OSError:
                continue
    return None


def resolve_camera_device(device: str | int) -> str | int:
    if isinstance(device, int):
        return device
    if str(device).isdigit():
        return int(str(device))
    path = Path(str(device)).expanduser()
    return str(path.resolve()) if path.exists() else str(path)


def discover_opencv_devices() -> list[dict[str, Any]]:
    """Discover OpenCV cameras using LeRobot's OpenCVCamera API when available."""

    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

        return OpenCVCamera.find_cameras()
    except Exception:
        if platform.system() == "Linux":
            return [{"id": str(path), "type": "OpenCV"} for path in sorted(Path("/dev").glob("video*"))]
        return [{"id": index, "type": "OpenCV"} for index in range(20)]


def configure_capture(capture: cv2.VideoCapture, width: int, height: int, fps: int, fourcc: str | None) -> None:
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    capture.set(cv2.CAP_PROP_FPS, float(fps))


def capture_sample(device: str | int, width: int, height: int, fps: int, fourcc: str | None) -> tuple[np.ndarray, dict[str, Any]] | None:
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        capture.release()
        return None
    configure_capture(capture, width, height, fps, fourcc)
    frame = None
    ok = False
    for _ in range(10):
        ok, frame = capture.read()
        if ok and frame is not None:
            break
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "backend": capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown",
    }
    capture.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), metadata


def probe_cameras(output_dir: Path, width: int, height: int, fps: int, fourcc: str | None, prefer_stable: bool) -> list[CameraCandidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CameraCandidate] = []
    seen: set[str] = set()
    for meta in discover_opencv_devices():
        device = meta.get("id")
        if device is None:
            continue
        device_key = str(device)
        if device_key in seen:
            continue
        seen.add(device_key)
        sample = capture_sample(device, width=width, height=height, fps=fps, fourcc=fourcc)
        if sample is None:
            continue
        image, capture_meta = sample
        stable = stable_camera_alias(device)
        selected: str | int = stable if prefer_stable and stable else device
        resolved = resolve_camera_device(selected)
        index = len(candidates)
        sample_path = output_dir / f"camera_{index:02d}.png"
        Image.fromarray(image).save(sample_path)
        candidates.append(
            CameraCandidate(
                index=index,
                device=device,
                selected_device=selected,
                resolved_device=resolved,
                sample_path=sample_path,
                width=int(capture_meta["width"]),
                height=int(capture_meta["height"]),
                fps=float(capture_meta["fps"]),
                backend=str(capture_meta["backend"]),
                stable_path=stable,
            )
        )
    write_contact_sheet(candidates, output_dir / "contact_sheet.jpg")
    return candidates


def write_contact_sheet(candidates: list[CameraCandidate], path: Path) -> None:
    if not candidates:
        return
    thumbs: list[Image.Image] = []
    for candidate in candidates:
        image = Image.open(candidate.sample_path).convert("RGB")
        image.thumbnail((320, 240))
        canvas = Image.new("RGB", (340, 290), "white")
        canvas.paste(image, (10, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 255), f"#{candidate.index}: {candidate.selected_device}", fill="black")
        thumbs.append(canvas)
    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 340, rows * 290), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 340, (idx // cols) * 290))
    sheet.save(path)


def display_candidates(candidates: list[CameraCandidate]) -> None:
    if not candidates:
        print("No cameras produced valid frames.")
        return
    for candidate in candidates:
        stable = f" stable={candidate.stable_path}" if candidate.stable_path else ""
        print(
            f"[{candidate.index}] selected={candidate.selected_device} raw={candidate.device}{stable}\n"
            f"    resolved={candidate.resolved_device}\n"
            f"    sample={candidate.sample_path}\n"
            f"    actual={candidate.width}x{candidate.height}@{candidate.fps:.2f} backend={candidate.backend}"
        )


def assign_roles(candidates: list[CameraCandidate], roles: list[str]) -> dict[str, str | int]:
    if not candidates:
        raise RuntimeError("No camera candidates to assign")
    mapping: dict[str, str | int] = {}
    by_index = {candidate.index: candidate for candidate in candidates}
    used_resolved: dict[str | int, str] = {}
    print("\nOpen the sample PNGs or contact_sheet.jpg to identify each camera view.")
    for role in roles:
        while True:
            answer = input(f"Camera index for {role.replace('_', ' ')}: ").strip()
            try:
                index = int(answer)
            except ValueError:
                print("Enter a numeric camera index.")
                continue
            if index not in by_index:
                print(f"Unknown camera index {index}.")
                continue
            candidate = by_index[index]
            if candidate.resolved_device in used_resolved:
                print(
                    f"Camera #{index} resolves to {candidate.resolved_device}, already assigned to "
                    f"{used_resolved[candidate.resolved_device]}. Pick a different physical camera."
                )
                continue
            used_resolved[candidate.resolved_device] = role
            mapping[DEFAULT_ROLE_TO_CONFIG_KEY[role]] = candidate.selected_device
            break
    return mapping


def write_yaml(path: Path, mapping: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(mapping, file, sort_keys=False)
    print(f"\nWrote {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("config/local_cameras.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_probe"))
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--prefer-raw", action="store_true", help="Write /dev/video* paths instead of stable aliases.")
    parser.add_argument("--list", action="store_true", help="Probe cameras, save samples, and exit without assignment.")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=sorted(DEFAULT_ROLE_TO_CONFIG_KEY),
        default=list(DEFAULT_ROLE_TO_CONFIG_KEY),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    candidates = probe_cameras(
        output_dir=args.output_dir,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        fourcc=args.fourcc,
        prefer_stable=not args.prefer_raw,
    )
    display_candidates(candidates)
    if args.list:
        return
    mapping = assign_roles(candidates, args.roles)
    mapping["camera_width"] = args.camera_width
    mapping["camera_height"] = args.camera_height
    mapping["camera_fps"] = args.camera_fps
    write_yaml(args.output, mapping)


if __name__ == "__main__":
    main()
