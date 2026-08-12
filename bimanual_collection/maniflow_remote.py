"""Remote ManiFlow policy runtime for Orbit live deployment."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from bimanual_collection.hardware.bimanual_robot import DEFAULT_JOINT_NAMES
from bimanual_collection.hardware.cameras import MatchedCameraFrame
from bimanual_collection.inference import ObservationSnapshot, bimanual_joint_vector


DEFAULT_MANIFLOW_CAMERAS = ("overhead", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class ManiFlowRemoteObservationAdapter:
    """Converts live Orbit observations into the remote ManiFlow server schema."""

    task_description: str
    cameras: tuple[str, ...] = DEFAULT_MANIFLOW_CAMERAS
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES

    @property
    def image_feature_keys(self) -> list[str]:
        return [f"observation.images.{camera}" for camera in self.cameras]

    def build(self, snapshot: ObservationSnapshot) -> dict[str, Any]:
        images: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            match = snapshot.camera_matches.get(camera)
            if match is None or match.frame is None:
                raise ValueError(f"Missing required camera frame: {camera}")
            if match.stale or match.missing or match.disconnected:
                raise ValueError(
                    f"Invalid camera frame for {camera}: "
                    f"stale={match.stale} missing={match.missing} disconnected={match.disconnected}"
                )
            images[camera] = _as_uint8_hwc_rgb(match)

        return {
            "agent_pos": bimanual_joint_vector(
                snapshot.follower_state.left,
                snapshot.follower_state.right,
                self.joint_names,
            ),
            "task_name": self.task_description,
            "images": images,
        }

    def build_history(self, snapshots: list[ObservationSnapshot]) -> dict[str, Any]:
        observations = [self.build(snapshot) for snapshot in snapshots]
        return {
            "agent_pos": np.stack([item["agent_pos"] for item in observations], axis=0),
            "task_name": self.task_description,
            "images": {
                camera: np.stack([item["images"][camera] for item in observations], axis=0)
                for camera in self.cameras
            },
        }


class ManiFlowRemotePolicyRuntime:
    """PolicyRuntime-compatible client for a localhost ManiFlow policy server."""

    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float = 10.0,
        cameras: tuple[str, ...] = DEFAULT_MANIFLOW_CAMERAS,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.cameras = tuple(cameras)
        self.image_feature_keys = [f"observation.images.{camera}" for camera in self.cameras]
        self.server_metadata: dict[str, Any] = {}
        self.n_obs_steps = 1
        self.latest_progress: float | None = None

    def load(self) -> None:
        self.server_metadata = self._get_json("/health")
        self.n_obs_steps = int(self.server_metadata.get("n_obs_steps", 1))
        server_cameras = tuple(self.server_metadata.get("cameras", ()))
        if server_cameras and server_cameras != self.cameras:
            raise RuntimeError(f"ManiFlow server cameras {server_cameras} do not match client {self.cameras}")

    def reset(self) -> None:
        self._post_bytes("/reset", b"")

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        _normalized_actions, actions = self.predict_action_chunk_with_debug(observation)
        return actions

    def predict_action_chunk_with_debug(self, observation: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray]:
        payload = self._encode_observation(observation)
        response = self._post_bytes("/predict", payload)
        with np.load(BytesIO(response), allow_pickle=False) as arrays:
            actions = np.asarray(arrays["actions"], dtype=np.float32)
            self.latest_progress = None
            if "progress" in arrays.files:
                progress = np.asarray(arrays["progress"], dtype=np.float32)
                if progress.size != 1:
                    raise ValueError(f"Expected scalar progress, got shape {progress.shape}")
                self.latest_progress = float(progress.reshape(-1)[0])
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        return None, actions

    def _encode_observation(self, observation: dict[str, Any]) -> bytes:
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("Remote ManiFlow observation must contain an images dict")

        fields: dict[str, np.ndarray] = {
            "agent_pos": np.asarray(observation["agent_pos"], dtype=np.float32),
            "task_name": np.asarray(str(observation["task_name"])),
        }
        for camera in self.cameras:
            if camera not in images:
                raise ValueError(f"Missing image for camera: {camera}")
            fields[f"image_{camera}"] = np.asarray(images[camera], dtype=np.uint8)

        buffer = BytesIO()
        np.savez_compressed(buffer, **fields)
        return buffer.getvalue()

    def _get_json(self, path: str) -> dict[str, Any]:
        import json

        response = self._request("GET", path)
        return json.loads(response.decode("utf-8"))

    def _post_bytes(self, path: str, body: bytes) -> bytes:
        return self._request("POST", path, body=body)

    def _request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        request = Request(
            self.server_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ManiFlow server {method} {path} failed: {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach ManiFlow server at {self.server_url}: {exc.reason}") from exc


def _as_uint8_hwc_rgb(match: MatchedCameraFrame) -> np.ndarray:
    if match.frame is None:
        raise ValueError(f"Missing frame for {match.camera_name}")
    image = np.asarray(match.frame.image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel image for {match.camera_name}, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)
