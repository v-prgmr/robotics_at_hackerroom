import torch
import torch.nn.functional as F

from maniflow.common.pytorch_util import dict_apply
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from maniflow.policy.topreward_loss import masked_weighted_mean


class TopRewardManiFlowTransformerImagePolicy(ManiFlowTransformerImagePolicy):
    """ManiFlow image policy with padding-aware per-action TOPReward weighting."""

    def __init__(self, *args, topreward_weighting="flow", **kwargs):
        super().__init__(*args, **kwargs)
        if topreward_weighting not in {"none", "flow", "both"}:
            raise ValueError("topreward_weighting must be one of: none, flow, both")
        self.topreward_weighting = topreward_weighting

    def compute_loss(self, batch, ema_model=None, **kwargs):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"]).to(self.device)
        action_valid = batch["action_valid_mask"].to(device=self.device, dtype=torch.bool)
        topreward_weight = batch["topreward_weight"].to(device=self.device, dtype=nactions.dtype)
        if not torch.isfinite(topreward_weight).all() or (topreward_weight <= 0).any():
            raise ValueError("TOPReward weights must be finite and positive")

        batch_size = nactions.shape[0]
        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get("task_name")
            assert lang_cond is not None, "Language goal is required"

        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)

        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)
        flow_slice = slice(0, flow_batchsize)
        consistency_slice = slice(flow_batchsize, flow_batchsize + consistency_batchsize)

        flow_target = self.get_flow_velocity(
            nactions[flow_slice],
            vis_cond=vis_cond[flow_slice],
            lang_cond=lang_cond[flow_slice] if lang_cond is not None else None,
        )
        v_flow_pred = self.model(
            sample=flow_target["x_t"],
            timestep=flow_target["t"].squeeze(),
            target_t=flow_target["target_t"].squeeze(),
            vis_cond=vis_cond[flow_slice],
            lang_cond=flow_target["lang_cond"] if lang_cond is not None else None,
        )

        consistency_target = self.get_consistency_velocity(
            nactions[consistency_slice],
            vis_cond=vis_cond[consistency_slice],
            lang_cond=lang_cond[consistency_slice] if lang_cond is not None else None,
            ema_model=ema_model,
        )
        v_ct_pred = self.model(
            sample=consistency_target["x_t"],
            timestep=consistency_target["t"].squeeze(),
            target_t=consistency_target["target_t"].squeeze(),
            vis_cond=vis_cond[consistency_slice],
            lang_cond=lang_cond[consistency_slice] if lang_cond is not None else None,
        )

        flow_elementwise = F.mse_loss(v_flow_pred, flow_target["v_target"], reduction="none")
        flow_unweighted = masked_weighted_mean(flow_elementwise, action_valid[flow_slice])
        flow_weight = topreward_weight[flow_slice] if self.topreward_weighting in {"flow", "both"} else None
        loss_flow = masked_weighted_mean(flow_elementwise, action_valid[flow_slice], flow_weight)

        ct_elementwise = F.mse_loss(v_ct_pred, consistency_target["v_target"], reduction="none")
        ct_unweighted = masked_weighted_mean(ct_elementwise, action_valid[consistency_slice])
        ct_weight = topreward_weight[consistency_slice] if self.topreward_weighting == "both" else None
        loss_ct = masked_weighted_mean(ct_elementwise, action_valid[consistency_slice], ct_weight)

        loss = loss_flow + loss_ct
        valid_weights = topreward_weight[action_valid]
        loss_dict = {
            "loss_flow": loss_flow.item(),
            "loss_flow_unweighted": flow_unweighted.item(),
            "loss_ct": loss_ct.item(),
            "loss_ct_unweighted": ct_unweighted.item(),
            "topreward_mean_weight": valid_weights.mean().item(),
            "topreward_min_weight": valid_weights.min().item(),
            "topreward_max_weight": valid_weights.max().item(),
            "v_flow_pred_magnitude": torch.sqrt(torch.mean(v_flow_pred**2)).item(),
            "v_ct_pred_magnitude": torch.sqrt(torch.mean(v_ct_pred**2)).item(),
            "bc_loss": loss.item(),
        }
        return loss, loss_dict
