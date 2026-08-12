import pathlib
import sys
import types

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "maniflow"))


class TopRewardManiFlowTransformerImagePolicy(nn.Module):
    pass


topreward_stub = types.ModuleType("maniflow.policy.topreward_maniflow_image_policy")
setattr(topreward_stub, "TopRewardManiFlowTransformerImagePolicy", TopRewardManiFlowTransformerImagePolicy)
sys.modules["maniflow.policy.topreward_maniflow_image_policy"] = topreward_stub

from maniflow.policy.maniflow_progress_value_policy import ManiFlowProgressValuePolicy

del sys.modules["maniflow.policy.topreward_maniflow_image_policy"]


class StubNormalizer:
    def normalize(self, obs):
        return obs


class StubObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 4)
        self.rgb_keys = ["left_wrist", "overhead", "right_wrist"]
        self.low_dim_keys = ["agent_pos"]
        self.key_shape_map = {key: (3, 2, 2) for key in self.rgb_keys}
        self.key_transform_map = nn.ModuleDict({key: nn.Identity() for key in self.rgb_keys})
        self.key_model_map = nn.ModuleDict({key: nn.Identity() for key in self.rgb_keys})
        self.key_projection_map = nn.ModuleDict(
            {key: nn.Linear(3, 4) for key in self.rgb_keys}
        )
        self.received_keys = None
        self.received_steps = None

    def forward(self, obs):
        self.received_keys = set(obs)
        self.received_steps = {key: value.shape[1] for key, value in obs.items()}
        batch_size = next(iter(obs.values())).shape[0]
        value = self.backbone(torch.ones(batch_size, 6, 1, device=self.backbone.weight.device))
        return value

    def aggregate_feature(self, feature):
        return feature.flatten(start_dim=2).transpose(1, 2)


class StubDiTX(nn.Module):
    language_encoder_out_dim = 3

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.text_calls = []

    def encode_text_input_T5(self, text, output_type, device):
        self.text_calls.append((text, output_type))
        return torch.ones(len(text), self.language_encoder_out_dim, device=device)


@pytest.fixture
def policy(monkeypatch):
    def stub_parent_init(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.obs_encoder = StubObsEncoder()
        self.model = StubDiTX()
        self.normalizer = StubNormalizer()
        self.obs_feature_dim = 4
        self.device = torch.device("cpu")

    monkeypatch.setattr(TopRewardManiFlowTransformerImagePolicy, "__init__", stub_parent_init)
    return ManiFlowProgressValuePolicy(progress_hidden_dim=5)


def observations(batch_size=2):
    return {
        "overhead": torch.zeros(batch_size, 3, 3, 2, 2),
        "left_wrist": torch.zeros(batch_size, 3, 3, 2, 2),
        "right_wrist": torch.zeros(batch_size, 3, 3, 2, 2),
        "agent_pos": torch.randn(batch_size, 3, 12),
        "task_name": ["task"] * batch_size,
    }


def test_head_shape_latest_rgb_only_and_freezing(policy):
    prediction = policy.predict_progress(observations())

    assert prediction.shape == (2, 1)
    assert torch.all(prediction >= -1) and torch.all(prediction <= 1)
    assert policy.obs_encoder.low_dim_keys == ["agent_pos"]
    assert policy.model.text_calls == [(["task", "task"], "sentence")]
    assert all(
        parameter.requires_grad == name.startswith("progress_head.")
        for name, parameter in policy.named_parameters()
    )


def test_frozen_modules_remain_eval_during_training(policy):
    policy.train()

    assert policy.training
    assert policy.progress_head.training
    assert not policy.obs_encoder.training
    assert not policy.model.training


def test_masked_progress_loss_and_zero_valid_batch(policy):
    batch = {
        "obs": observations(3),
        "progress_target": torch.tensor([0.5, -0.5, 100.0]),
        "progress_valid": torch.tensor([True, True, False]),
    }
    prediction = policy.predict_progress(batch["obs"]).detach().reshape(-1)
    expected = ((prediction[:2] - batch["progress_target"][:2]) ** 2).mean()

    loss, metrics = policy.compute_loss(batch)
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["progress_valid_count"] == 2

    batch["progress_valid"] = torch.zeros(3, dtype=torch.bool)
    zero_loss, zero_metrics = policy.compute_loss(batch)
    zero_loss.backward()
    assert zero_loss.item() == 0.0
    assert zero_metrics["progress_valid_count"] == 0
    assert all(parameter.grad is not None for parameter in policy.progress_head.parameters())


def test_source_loader_allows_exactly_missing_progress_head(policy):
    source = {
        key: value.clone()
        for key, value in policy.state_dict().items()
        if not key.startswith("progress_head.")
    }
    incompatible = policy.load_source_state_dict(source)
    assert set(incompatible.missing_keys) == {
        key for key in policy.state_dict() if key.startswith("progress_head.")
    }

    with pytest.raises(RuntimeError, match="must match exactly"):
        policy.load_source_state_dict({key: value for key, value in source.items() if key != "model.backbone.bias"})
    with pytest.raises(RuntimeError, match="must match exactly"):
        policy.load_source_state_dict(source | {"unexpected.weight": torch.ones(1)})


def test_optimizer_step_changes_only_progress_head_and_has_nonzero_head_gradient(policy):
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if not name.startswith("progress_head.")
    }
    optimizer = torch.optim.SGD(policy.progress_head.parameters(), lr=0.1)
    batch = {
        "obs": observations(2),
        "progress_target": torch.tensor([0.0, 1.0]),
        "progress_valid": torch.tensor([True, True]),
    }

    loss, _ = policy.compute_loss(batch)
    loss.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in policy.progress_head.parameters()
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in policy.named_parameters()
        if not name.startswith("progress_head.")
    )
    optimizer.step()
    for name, before in frozen_before.items():
        torch.testing.assert_close(dict(policy.named_parameters())[name], before, rtol=0, atol=0)
