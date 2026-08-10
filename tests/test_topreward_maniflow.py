import json
import math

import numpy as np
import pandas as pd
import pytest

from maniflow_orbit_bridge.convert_orbit_lerobot_to_maniflow import (
    _load_topreward_episode,
    build_arg_parser,
    compute_topreward_action_arrays,
)
from maniflow_orbit_bridge.convert_orbit_topreward_to_maniflow import (
    DEFAULT_CAMERAS,
    _batch_ranges,
    _completed_episode_indexes,
    build_arg_parser as build_full_arg_parser,
    discover_episodes,
)


def _anchors(indices, scores):
    return [
        {"anchor_index": index, "anchor_timestep_index": timestep, "logp_true": score}
        for index, (timestep, score) in enumerate(zip(indices, scores, strict=True))
    ]


def test_converter_defaults_to_revised_paper_weighting():
    args = build_arg_parser().parse_args(["--lerobot-root", "input", "--output-zarr", "output"])

    assert args.topreward_score_mode == "raw_logp_true"
    assert args.topreward_exponent_scale is None
    assert args.topreward_max_weight == 2.0


def test_full_converter_output_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_full_arg_parser().parse_args(
            [
                "--orbit-root",
                "input",
                "--topreward-results",
                "scores",
                "--output-zarr",
                "output",
                "--overwrite",
                "--resume",
            ]
        )


def test_image_batches_align_interior_writes_to_zarr_chunks():
    ranges = list(_batch_ranges(total=600, global_start=10, batch_size=256, chunk_length=64))

    assert ranges == [(0, 54), (54, 310), (310, 566), (566, 600)]
    assert all((10 + start) % 64 == 0 for start, _ in ranges[1:])


def test_resume_state_accepts_interrupted_matching_conversion():
    config = {"version": 1, "image_size": 224}
    state = {"status": "in_progress", "config": config, "completed_episodes": [0, 2]}

    assert _completed_episode_indexes(state, config, episode_count=3) == {0, 2}


def test_resume_state_rejects_incompatible_or_invalid_state():
    config = {"version": 1, "image_size": 224}

    with pytest.raises(ValueError, match="configuration"):
        _completed_episode_indexes(
            {"config": {**config, "image_size": 128}, "completed_episodes": [0]},
            config,
            episode_count=1,
        )
    with pytest.raises(ValueError, match="episode index"):
        _completed_episode_indexes(
            {"config": config, "completed_episodes": [1]}, config, episode_count=1
        )


def test_full_converter_defaults_to_all_collections_and_raw_weighting():
    args = build_full_arg_parser().parse_args(
        ["--orbit-root", "input", "--topreward-results", "scores", "--output-zarr", "output"]
    )

    assert args.collection is None
    assert args.topreward_score_mode == "raw_logp_true"
    assert args.topreward_exponent_scale is None
    assert args.topreward_max_weight == 2.0
    assert args.video_workers == 3
    assert args.write_batch_size == 256
    assert args.compression_level == 1
    assert args.resume is False


def test_full_converter_preserves_nonzero_source_episode_ids(tmp_path):
    orbit_root = tmp_path / "orbit"
    score_root = tmp_path / "scores"
    episode = orbit_root / "maniflow_hil_bounded" / "episode-000004"
    (episode / "videos").mkdir(parents=True)
    for camera in DEFAULT_CAMERAS:
        (episode / "videos" / f"{camera}.mp4").touch()
    (episode / "episode_metadata.json").write_text(
        json.dumps({"episode_id": "episode-000004", "task_description": "test"}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "timestep_index": [0, 1],
            "left_follower_joints": [np.zeros(6, dtype=np.float32)] * 2,
            "right_follower_joints": [np.zeros(6, dtype=np.float32)] * 2,
            "left_commanded_action": [np.zeros(6, dtype=np.float32)] * 2,
            "right_commanded_action": [np.zeros(6, dtype=np.float32)] * 2,
            **{f"{camera}_video_frame_index": [0, 1] for camera in DEFAULT_CAMERAS},
        }
    ).to_parquet(episode / "timesteps.parquet")
    score_path = score_root / "episodes" / "maniflow_hil_bounded" / "episode-000004.json"
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": "maniflow_hil_bounded",
                "episode_id": "episode-000004",
                "num_timesteps": 2,
                "anchors": _anchors([0, 1], [-2.0, -1.0]),
            }
        ),
        encoding="utf-8",
    )

    specs = discover_episodes(
        orbit_root=orbit_root,
        topreward_results=score_root,
        collections=None,
        cameras=DEFAULT_CAMERAS,
        frame_stride=1,
        max_episodes=None,
        score_mode="raw_logp_true",
        exponent_scale=0.2,
        max_weight=2.0,
    )

    assert [(spec.collection, spec.episode_id, spec.source_episode_index) for spec in specs] == [
        ("maniflow_hil_bounded", "episode-000004", 4)
    ]


def test_normalized_progress_broadcasts_over_causal_action_slices():
    delta, unclipped, weight = compute_topreward_action_arrays(
        total_frames=6,
        anchors=_anchors([0, 2, 5], [0.0, 1.0, 0.5]),
        score_mode="normalized_progress",
        exponent_scale=2.0,
        max_weight=2.0,
    )

    np.testing.assert_allclose(delta, [1.0, 1.0, -0.5, -0.5, -0.5, 0.0])
    np.testing.assert_allclose(unclipped, [math.exp(2.0), math.exp(2.0), math.exp(-1.0), math.exp(-1.0), math.exp(-1.0), 1.0])
    np.testing.assert_allclose(weight, [2.0, 2.0, math.exp(-1.0), math.exp(-1.0), math.exp(-1.0), 1.0])
    assert weight[2] < 1.0


def test_flat_progress_and_unmeasured_actions_remain_neutral():
    delta, unclipped, weight = compute_topreward_action_arrays(
        total_frames=4,
        anchors=_anchors([0, 3], [-2.0, -2.0]),
        score_mode="normalized_progress",
        exponent_scale=2.0,
        max_weight=2.0,
    )

    np.testing.assert_array_equal(delta, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(unclipped, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(weight, np.ones(4, dtype=np.float32))


def test_raw_logp_ablation_uses_configured_beta_without_lower_clamp():
    _, unclipped, weight = compute_topreward_action_arrays(
        total_frames=3,
        anchors=_anchors([0, 2], [-1.0, -3.0]),
        score_mode="raw_logp_true",
        exponent_scale=0.2,
        max_weight=2.0,
    )

    np.testing.assert_allclose(unclipped[:2], math.exp(-0.4))
    np.testing.assert_allclose(weight[:2], math.exp(-0.4))


def test_topreward_anchors_must_cover_observation_endpoints():
    with pytest.raises(ValueError, match="cover the episode observation endpoints"):
        compute_topreward_action_arrays(
            total_frames=5,
            anchors=_anchors([1, 4], [0.0, 1.0]),
            score_mode="normalized_progress",
            exponent_scale=2.0,
            max_weight=2.0,
        )


def test_topreward_episode_loader_validates_provenance_and_length(tmp_path):
    path = tmp_path / "episodes" / "expert" / "episode-000001.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": "expert",
                "episode_id": "episode-000001",
                "num_timesteps": 10,
                "anchors": [],
            }
        ),
        encoding="utf-8",
    )

    assert _load_topreward_episode(tmp_path, "expert", 0, 10)["episode_id"] == "episode-000001"
    with pytest.raises(ValueError, match="frame count mismatch"):
        _load_topreward_episode(tmp_path, "expert", 0, 9)


def test_masked_weighted_mean_excludes_padding_without_weight_normalization():
    torch = pytest.importorskip("torch")
    from maniflow_orbit_bridge.maniflow_policy.topreward_loss import masked_weighted_mean

    elementwise = torch.ones((1, 3, 2))
    valid = torch.tensor([[True, True, False]])
    weights = torch.tensor([[2.0, 0.5, 100.0]])

    assert masked_weighted_mean(elementwise, valid).item() == pytest.approx(1.0)
    assert masked_weighted_mean(elementwise, valid, weights).item() == pytest.approx(1.25)
    with pytest.raises(ValueError, match="no valid actions"):
        masked_weighted_mean(elementwise, torch.zeros_like(valid), weights)
