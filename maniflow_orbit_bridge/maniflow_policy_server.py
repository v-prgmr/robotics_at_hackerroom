"""Localhost ManiFlow policy server for live Orbit robot deployment.

Run this in the ManiFlow conda environment. The Orbit robot client runs in the
normal Orbit environment and talks to this server over localhost.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections import deque
from datetime import datetime
from importlib.util import find_spec
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


DEFAULT_CAMERAS = ("overhead", "left_wrist", "right_wrist")


def add_maniflow_to_path(maniflow_dir: Path | None) -> Path | None:
    """Make a ManiFlow checkout importable for Hydra-instantiated checkpoint targets."""

    candidates = []
    if maniflow_dir is not None:
        candidates.append(maniflow_dir.expanduser())
    candidates.extend(
        [
            Path.cwd() / "maniflow",
            Path(__file__).resolve().parents[1] / "maniflow",
            Path("/workspace/maniflow"),
        ]
    )

    for candidate in candidates:
        package_dir = candidate / "maniflow"
        if package_dir.exists():
            sys.path.insert(0, str(candidate.resolve()))
            return candidate.resolve()
    return None


class ManiFlowPolicyService:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        prefer_ema: bool = True,
    ) -> None:
        self.checkpoint = checkpoint.expanduser()
        self.device = torch.device(device)
        self.cameras = tuple(cameras)
        self.prefer_ema = bool(prefer_ema)
        self.policy: Any | None = None
        self.cfg: Any | None = None
        self.n_obs_steps = 1
        self.n_action_steps: int | None = None
        self.action_dim: int | None = None
        self._history: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)
        payload = torch.load(self.checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        self.cfg = cfg
        OmegaConf.resolve(cfg)

        policy = hydra.utils.instantiate(cfg.policy)
        state_dicts = payload.get("state_dicts", {})
        state_key = "ema_model" if self.prefer_ema and "ema_model" in state_dicts else "model"
        if state_key not in state_dicts:
            raise KeyError(f"Checkpoint has no '{state_key}' state dict; available keys: {sorted(state_dicts)}")
        policy.load_state_dict(state_dicts[state_key])
        policy.to(self.device)
        policy.eval()

        self.policy = policy
        self.n_obs_steps = int(getattr(policy, "n_obs_steps", cfg.n_obs_steps))
        self.n_action_steps = int(getattr(policy, "n_action_steps", cfg.n_action_steps))
        self.action_dim = int(getattr(policy, "action_dim", cfg.shape_meta.action.shape[0]))
        self._history = deque(maxlen=self.n_obs_steps)
        logger.info("Loaded %s from %s on %s", state_key, self.checkpoint, self.device)

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "cameras": list(self.cameras),
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.action_dim,
            "loaded": self.policy is not None,
            "supports_progress": callable(getattr(self.policy, "predict_progress", None)),
        }

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def predict(self, payload_bytes: bytes) -> np.ndarray:
        return self.predict_with_progress(payload_bytes)["actions"]

    def predict_with_progress(self, payload_bytes: bytes) -> dict[str, np.ndarray]:
        if self.policy is None:
            raise RuntimeError("Policy is not loaded")

        samples = self._decode_payload(payload_bytes)
        with self._lock:
            if len(samples) > 1:
                self._history.clear()
                self._history.extend(samples[-self.n_obs_steps :])
            else:
                self._history.append(samples[0])
            history = list(self._history)
            while len(history) < self.n_obs_steps:
                history.insert(0, history[0])
            obs_dict = self._history_to_obs(history[-self.n_obs_steps :])

        with torch.inference_mode():
            result = self.policy.predict_action(obs_dict)
            progress = None
            if callable(getattr(self.policy, "predict_progress", None)):
                progress = self.policy.predict_progress(obs_dict)
        actions = result["action"].detach().cpu().numpy().astype(np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        response = {"actions": actions}
        if progress is not None:
            response["progress"] = progress.detach().cpu().numpy().astype(np.float32)
        return response

    def _decode_payload(self, payload_bytes: bytes) -> list[dict[str, Any]]:
        with np.load(BytesIO(payload_bytes), allow_pickle=False) as arrays:
            agent_pos = np.asarray(arrays["agent_pos"], dtype=np.float32)
            if agent_pos.ndim == 1:
                agent_pos = agent_pos[None, :]
            if agent_pos.ndim != 2 or agent_pos.shape[1] != 12:
                raise ValueError(f"Expected agent_pos shape (T, 12), got {agent_pos.shape}")
            task_name = str(np.asarray(arrays["task_name"]).item())
            camera_images: dict[str, np.ndarray] = {}
            for camera in self.cameras:
                images = np.asarray(arrays[f"image_{camera}"])
                if images.ndim == 3:
                    images = images[None, ...]
                if len(images) != len(agent_pos):
                    raise ValueError(
                        f"Camera {camera} history length {len(images)} does not match agent_pos {len(agent_pos)}"
                    )
                camera_images[camera] = images
        return [
            {
                "agent_pos": agent_pos[index],
                "task_name": task_name,
                "images": {
                    camera: self._image_to_chw_uint8(camera_images[camera][index], camera)
                    for camera in self.cameras
                },
            }
            for index in range(len(agent_pos))
        ]

    @staticmethod
    def _image_to_chw_uint8(image: np.ndarray, camera: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image for {camera}, got shape {image.shape}")
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        if image.shape[0] != 3:
            raise ValueError(f"Expected 3-channel image for {camera}, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _history_to_obs(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "agent_pos": torch.from_numpy(np.stack([item["agent_pos"] for item in history], axis=0))[None].to(
                self.device
            ),
            "task_name": [history[-1]["task_name"]],
        }
        for camera in self.cameras:
            images = np.stack([item["images"][camera] for item in history], axis=0)
            obs[camera] = torch.from_numpy(images)[None].to(self.device, dtype=torch.float32) / 255.0
        return obs


class PolicyRequestHandler(BaseHTTPRequestHandler):
    service: ManiFlowPolicyService

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(self.service.metadata())
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/reset":
                self.service.reset()
                self._send_json({"ok": True})
                return
            if self.path == "/predict":
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                result = self.service.predict_with_progress(payload)
                buffer = BytesIO()
                np.savez_compressed(buffer, **result)
                self._send_bytes(buffer.getvalue(), content_type="application/octet-stream")
                return
            self.send_error(404, "unknown endpoint")
        except Exception as exc:
            logger.exception("Request failed")
            self.send_error(500, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), content_type="application/json")

    def _send_bytes(self, payload: bytes, *, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a ManiFlow .ckpt file")
    parser.add_argument(
        "--maniflow-dir",
        type=Path,
        help="Path to the ManiFlow checkout. Defaults to ./maniflow, repo-local maniflow, or /workspace/maniflow.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera", action="append", dest="cameras", help="Camera key; defaults to overhead/left_wrist/right_wrist")
    parser.add_argument("--no-ema", action="store_true", help="Use model weights instead of ema_model when both exist")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    maniflow_dir = add_maniflow_to_path(args.maniflow_dir)
    if find_spec("maniflow") is None:
        raise ModuleNotFoundError(
            "Could not import 'maniflow'. Pass --maniflow-dir /path/to/maniflow or run from a checkout "
            "that has ./maniflow. If using the local checkout, run "
            "`python maniflow_orbit_bridge/install_into_maniflow.py --maniflow-dir ./maniflow --overwrite` first."
        )
    if maniflow_dir is not None:
        logger.info("Using ManiFlow checkout: %s", maniflow_dir)
    service = ManiFlowPolicyService(
        args.checkpoint,
        device=args.device,
        cameras=tuple(args.cameras or DEFAULT_CAMERAS),
        prefer_ema=not args.no_ema,
    )
    service.load()
    PolicyRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), PolicyRequestHandler)
    logger.info("Serving ManiFlow policy at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping ManiFlow policy server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
