import json

import numpy as np
import pandas as pd

from topreward_orbit.score_orbit_dataset import (
    aggregate_results,
    build_arg_parser,
    build_prefix_specs,
    classify_collection,
    minmax_for_plot,
    uniformly_spaced_indices,
)


def test_batch_size_defaults_to_low_vram_mode():
    args = build_arg_parser().parse_args(["--input-root", "input", "--output-dir", "output"])
    assert args.batch_size == 1

    args = build_arg_parser().parse_args(
        ["--input-root", "input", "--output-dir", "output", "--batch-size", "2"]
    )
    assert args.batch_size == 2


def test_prefix_anchors_and_prefix_frames_are_independent():
    timesteps = pd.DataFrame(
        {
            "timestep_index": np.arange(101),
            "overhead_video_frame_index": np.arange(101) * 2,
        }
    )

    specs = build_prefix_specs(timesteps, camera="overhead", num_anchors=15, max_frames=16)

    assert len(specs) == 15
    assert specs[0].anchor_timestep_index == 0
    assert specs[-1].anchor_timestep_index == 100
    assert specs[-1].sampled_timestep_indices[0] == 0
    assert specs[-1].sampled_timestep_indices[-1] == 100
    assert len(specs[-1].sampled_timestep_indices) == 16
    assert specs[-1].sampled_video_frame_indices[-1] == 200
    assert all(len(spec.sampled_timestep_indices) <= 16 for spec in specs)


def test_uniform_indices_include_interval_ends():
    assert uniformly_spaced_indices(0, 100, 4) == [0, 33, 67, 100]
    assert uniformly_spaced_indices(3, 5, 10) == [3, 4, 5]


def test_minmax_is_only_episode_local_math():
    np.testing.assert_allclose(minmax_for_plot([-4.0, -2.0, 0.0]), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(minmax_for_plot([-2.0, -2.0]), [1.0, 1.0])


def test_hil_success_is_not_policy_success():
    assert classify_collection("anything", {"collection_type": "hil", "success": True}) == "hil_correction"
    assert (
        classify_collection(
            "rollouts", {"episode_type": "autonomous_rollout", "terminal_success": True}
        )
        == "policy_success"
    )
    assert (
        classify_collection(
            "rollouts", {"episode_type": "autonomous_rollout", "terminal_success": False}
        )
        == "policy_failure"
    )
    assert classify_collection("expert_successes", {"success": True}) == "expert"


def test_missing_camera_indices_use_nearest_available_frame_for_scoring():
    timesteps = pd.DataFrame(
        {
            "timestep_index": [0, 1, 2, 3],
            "overhead_video_frame_index": [0, np.nan, np.nan, 2],
        }
    )

    specs = build_prefix_specs(timesteps, camera="overhead", num_anchors=4, max_frames=4)

    assert [spec.anchor_video_frame_index for spec in specs] == [0, 0, 0, 2]


def _payload(dataset, episode_id, collection_type, prefix_change, yes_no_margin, terminal_success):
    return {
        "status": "complete",
        "dataset": dataset,
        "episode_id": episode_id,
        "collection_type": collection_type,
        "terminal_success": terminal_success,
        "metadata_success": terminal_success,
        "task": "Do the task.",
        "num_timesteps": 3,
        "timestep_indices": [0, 1, 2],
        "timestep_video_frame_indices": [0, 0, 1],
        "camera": "overhead",
        "model_name": "test/model",
        "num_prefix_anchors": 2,
        "max_frames_per_prefix": 16,
        "anchors": [
            {
                "anchor_index": 0,
                "anchor_timestep_index": 0,
                "anchor_video_frame_index": 0,
                "sampled_timestep_indices": [0],
                "sampled_video_frame_indices": [0],
                "sampled_frame_count": 1,
                "logp_true": -4.0,
                "logp_false": -1.0,
                "true_false_margin": -3.0,
                "normalized_true_for_plot": 0.0,
                "normalized_margin_for_plot": 0.0,
            },
            {
                "anchor_index": 1,
                "anchor_timestep_index": 2,
                "anchor_video_frame_index": 1,
                "sampled_timestep_indices": [0, 2],
                "sampled_video_frame_indices": [0, 1],
                "sampled_frame_count": 2,
                "logp_true": -1.0,
                "logp_false": -3.0,
                "true_false_margin": 2.0,
                "normalized_true_for_plot": 1.0,
                "normalized_margin_for_plot": 1.0,
            },
        ],
        "terminal_logp_true": -1.0,
        "terminal_logp_false": -3.0,
        "terminal_true_false_margin": 2.0,
        "prefix_margin_change": prefix_change,
        "full_video_logp_yes": -1.0,
        "full_video_logp_no": -3.0,
        "full_video_yes_no_margin": yes_no_margin,
        "voc_true": 1.0,
        "voc_margin": 1.0,
    }


def test_aggregate_standardizes_success_only_on_policy_episodes(tmp_path):
    payloads = [
        _payload("successes", "episode-000001", "policy_success", 3.0, 4.0, True),
        _payload("failures", "episode-000001", "policy_failure", -1.0, -2.0, False),
        _payload("hil", "episode-000001", "hil_correction", 100.0, 100.0, None),
    ]
    for payload in payloads:
        path = tmp_path / "episodes" / payload["dataset"] / f"{payload['episode_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    episode_path, anchor_path, timestep_path = aggregate_results(tmp_path)

    episodes = pd.read_parquet(episode_path).set_index("collection_type")
    assert episodes.loc["policy_success", "success_score"] == 2.0
    assert episodes.loc["policy_failure", "success_score"] == -2.0
    assert np.isnan(episodes.loc["hil_correction", "success_score"])
    assert len(pd.read_parquet(anchor_path)) == 6
    timesteps = pd.read_parquet(timestep_path)
    assert len(timesteps) == 9
    assert timesteps.iloc[1]["video_frame_index"] == 0
    assert timesteps.iloc[1]["topreward_logp_true"] == -2.5
