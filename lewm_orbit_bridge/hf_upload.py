"""Hugging Face Hub configuration and retrying upload helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


@dataclass(frozen=True)
class HuggingFaceUploadConfig:
    enabled: bool
    repo_id: str = ""
    token: str = field(default="", repr=False)
    private: bool = True
    checkpoint_interval_epochs: int = 5
    remote_prefix: str = ""
    retries: int = 3
    retry_delay_s: int = 10

    @classmethod
    def from_environment(
        cls, experiment_name: str, environment: Mapping[str, str] | None = None
    ) -> "HuggingFaceUploadConfig":
        env = os.environ if environment is None else environment
        enabled = _parse_bool(env.get("HF_UPLOAD_ENABLED", "false"), "HF_UPLOAD_ENABLED")
        interval = _parse_nonnegative_int(
            env.get("HF_CHECKPOINT_INTERVAL_EPOCHS", "5"), "HF_CHECKPOINT_INTERVAL_EPOCHS"
        )
        retries = _parse_nonnegative_int(env.get("HF_UPLOAD_RETRIES", "3"), "HF_UPLOAD_RETRIES")
        retry_delay_s = _parse_nonnegative_int(
            env.get("HF_UPLOAD_RETRY_DELAY_S", "10"), "HF_UPLOAD_RETRY_DELAY_S"
        )
        private = _parse_bool(env.get("HF_PRIVATE", "true"), "HF_PRIVATE")
        repo_id = env.get("HF_REPO_ID", "").strip()
        token = (env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        remote_prefix = env.get("HF_REMOTE_PREFIX", f"runs/{experiment_name}").strip().strip("/")
        if enabled:
            if not repo_id:
                raise ValueError("HF_REPO_ID is required when HF_UPLOAD_ENABLED=true")
            if not token:
                raise ValueError(
                    "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required when HF_UPLOAD_ENABLED=true"
                )
            if not remote_prefix:
                raise ValueError("HF_REMOTE_PREFIX cannot be empty when HF upload is enabled")
            if retries < 1:
                raise ValueError("HF_UPLOAD_RETRIES must be at least 1 when HF upload is enabled")
        return cls(
            enabled=enabled,
            repo_id=repo_id,
            token=token,
            private=private,
            checkpoint_interval_epochs=interval,
            remote_prefix=remote_prefix,
            retries=retries,
            retry_delay_s=retry_delay_s,
        )

    def public_metadata(self) -> dict[str, object]:
        """Return reproducibility metadata without exposing the Hub token."""

        return {
            "enabled": self.enabled,
            "repo_id": self.repo_id if self.enabled else None,
            "private": self.private if self.enabled else None,
            "checkpoint_interval_epochs": self.checkpoint_interval_epochs,
            "remote_prefix": self.remote_prefix if self.enabled else None,
        }

    def should_upload_checkpoint(self, epoch: int) -> bool:
        return (
            self.enabled
            and self.checkpoint_interval_epochs > 0
            and epoch > 0
            and epoch % self.checkpoint_interval_epochs == 0
        )


class HuggingFaceRunUploader:
    """Create a model repository and upload recovery or final artifacts."""

    def __init__(self, config: HuggingFaceUploadConfig) -> None:
        if not config.enabled:
            raise ValueError("Cannot create an uploader when HF upload is disabled")
        from huggingface_hub import HfApi, create_repo

        self.config = config
        self.api = HfApi(token=config.token)

        def initialize_repo() -> None:
            create_repo(
                config.repo_id,
                repo_type="model",
                private=config.private,
                exist_ok=True,
                token=config.token,
            )

        self._retry("repository initialization", initialize_repo)
        info_result = []
        self._retry(
            "repository visibility check",
            lambda: info_result.append(
                self.api.repo_info(repo_id=config.repo_id, repo_type="model", token=config.token)
            ),
        )
        info = info_result[0]
        self.actual_private = bool(info.private)
        if config.private and not self.actual_private:
            raise ValueError(
                f"HF_PRIVATE=true but existing repository {config.repo_id!r} is public; refusing upload"
            )

    def _retry(self, label: str, operation) -> None:
        for attempt in range(1, self.config.retries + 1):
            try:
                operation()
                return
            except Exception:
                if attempt == self.config.retries:
                    raise
                print(
                    f"Hugging Face {label} failed (attempt {attempt}/{self.config.retries}); "
                    f"retrying in {self.config.retry_delay_s}s",
                    flush=True,
                )
                time.sleep(self.config.retry_delay_s)

    def upload_recovery_checkpoint(
        self,
        checkpoint: Path,
        epoch: int,
        global_step: int,
        best_checkpoint: Path | None = None,
    ) -> None:
        path_in_repo = f"{self.config.remote_prefix}/recovery/last.ckpt"
        self._retry(
            "recovery checkpoint upload",
            lambda: self.api.upload_file(
                path_or_fileobj=str(checkpoint),
                path_in_repo=path_in_repo,
                repo_id=self.config.repo_id,
                repo_type="model",
                token=self.config.token,
                commit_message=f"Update LeWM recovery checkpoint at epoch {epoch}, step {global_step}",
            ),
        )
        print(f"Uploaded resumable checkpoint to hf://{self.config.repo_id}/{path_in_repo}", flush=True)
        if best_checkpoint is not None and best_checkpoint.is_file():
            best_path_in_repo = f"{self.config.remote_prefix}/recovery/best.ckpt"
            self._retry(
                "best checkpoint upload",
                lambda: self.api.upload_file(
                    path_or_fileobj=str(best_checkpoint),
                    path_in_repo=best_path_in_repo,
                    repo_id=self.config.repo_id,
                    repo_type="model",
                    token=self.config.token,
                    commit_message=f"Update best LeWM checkpoint through epoch {epoch}",
                ),
            )
            print(
                f"Uploaded best checkpoint to hf://{self.config.repo_id}/{best_path_in_repo}",
                flush=True,
            )

    def upload_final_run(self, output_dir: Path) -> None:
        self._retry(
            "final run upload",
            lambda: self.api.upload_folder(
                folder_path=str(output_dir),
                path_in_repo=self.config.remote_prefix,
                repo_id=self.config.repo_id,
                repo_type="model",
                token=self.config.token,
                commit_message="Upload completed Orbit LeWM experiment",
            ),
        )
        print(
            f"Uploaded final run to hf://{self.config.repo_id}/{self.config.remote_prefix}", flush=True
        )
