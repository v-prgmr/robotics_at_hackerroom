from __future__ import annotations

import pathlib

import dill
import torch
import torch.nn.functional as F
from torch import nn

from maniflow.policy.topreward_maniflow_image_policy import TopRewardManiFlowTransformerImagePolicy


class ManiFlowProgressValuePolicy(TopRewardManiFlowTransformerImagePolicy):
    """Frozen ManiFlow visual-language backbone with a scalar progress head."""

    def __init__(
        self,
        *args,
        progress_hidden_dim=512,
        progress_rgb_keys=("overhead", "left_wrist", "right_wrist"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_rgb_keys = tuple(progress_rgb_keys)
        if len(self.progress_rgb_keys) != 3:
            raise ValueError("progress_rgb_keys must contain exactly three RGB observation keys")
        encoder_rgb_keys = set(self.obs_encoder.rgb_keys)
        if set(self.progress_rgb_keys) != encoder_rgb_keys:
            raise ValueError(
                f"progress_rgb_keys must match the encoder RGB keys: {sorted(encoder_rgb_keys)}"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        text_dim = self.model.language_encoder_out_dim
        self.progress_head = nn.Sequential(
            nn.Linear(self.obs_feature_dim + text_dim, progress_hidden_dim),
            nn.SiLU(),
            nn.Linear(progress_hidden_dim, progress_hidden_dim),
            nn.SiLU(),
            nn.Linear(progress_hidden_dim, 1),
            nn.Tanh(),
        )
        self._set_frozen_modules_eval()

    def _set_frozen_modules_eval(self):
        for name, module in self.named_children():
            if name != "progress_head":
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._set_frozen_modules_eval()
        self.progress_head.train(mode)
        return self

    def _normalized_rgb_observations(self, obs):
        missing = [key for key in self.progress_rgb_keys if key not in obs]
        if missing:
            raise KeyError(f"Missing progress RGB observations: {missing}")
        rgb_obs = {key: obs[key] for key in self.progress_rgb_keys}
        rgb_obs = self.normalizer.normalize(rgb_obs)
        return {key: value[:, -1:, ...].to(self.device) for key, value in rgb_obs.items()}

    def _encode_visual_tokens(self, obs):
        embeddings = []
        for key in self.obs_encoder.rgb_keys:
            image = obs[key]
            if image.max() > 1.0:
                image = image / 255.0
            if image.shape[-1] == 3:
                image = image.permute(0, 1, 4, 2, 3)
            batch_size, steps = image.shape[:2]
            image = image.reshape(batch_size * steps, *image.shape[2:])
            target_shape = self.obs_encoder.key_shape_map[key]
            if image.shape[1:] != target_shape:
                image = F.interpolate(image, size=target_shape[1:], mode="bilinear", align_corners=False)
            image = self.obs_encoder.key_transform_map[key](image).to(self.device)
            feature = self.obs_encoder.key_model_map[key](image).to(self.device)
            feature = self.obs_encoder.aggregate_feature(feature)
            embedding = self.obs_encoder.key_projection_map[key](feature)
            embeddings.append(embedding.reshape(batch_size, -1, self.obs_feature_dim))
        return torch.cat(embeddings, dim=1)

    def predict_progress(self, obs):
        task_name = obs.get("task_name")
        if task_name is None:
            raise KeyError("Progress prediction requires obs['task_name']")

        with torch.no_grad():
            visual_tokens = self._encode_visual_tokens(self._normalized_rgb_observations(obs))
            visual_embedding = visual_tokens.mean(dim=1)
            text_embedding = self.model.encode_text_input_T5(
                task_name, output_type="sentence", device=self.device
            )
        head_weight = self.progress_head[0].weight
        features = torch.cat([visual_embedding, text_embedding], dim=-1).to(
            device=head_weight.device, dtype=head_weight.dtype
        )
        return self.progress_head(features)

    def compute_loss(self, batch, ema_model=None, **kwargs):
        prediction = self.predict_progress(batch["obs"])
        target = batch["progress_target"].to(device=prediction.device, dtype=prediction.dtype).reshape(-1, 1)
        valid = batch["progress_valid"].to(device=prediction.device, dtype=torch.bool).reshape(-1)
        if target.shape != prediction.shape or valid.shape[0] != prediction.shape[0]:
            raise ValueError("progress_target and progress_valid must contain one value per batch item")

        valid_count = int(valid.sum().item())
        if valid_count == 0:
            loss = prediction.sum() * 0.0
            return loss, {
                "progress_loss": 0.0,
                "progress_mae": 0.0,
                "progress_valid_count": 0,
            }

        error = prediction[valid] - target[valid]
        loss = error.square().mean()
        return loss, {
            "progress_loss": loss.item(),
            "progress_mae": error.abs().mean().item(),
            "progress_valid_count": valid_count,
        }

    def load_source_ema_checkpoint(self, path, state_key="ema_model"):
        checkpoint_path = pathlib.Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Source checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
        state_dicts = payload.get("state_dicts", {})
        if state_key not in state_dicts:
            raise KeyError(f"Checkpoint has no '{state_key}' state dict; available keys: {sorted(state_dicts)}")
        return self.load_source_state_dict(state_dicts[state_key])

    def load_source_state_dict(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {key for key in self.state_dict() if key.startswith("progress_head.")}
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != allowed_missing or unexpected:
            raise RuntimeError(
                "Source checkpoint must match exactly except for progress_head; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return incompatible
