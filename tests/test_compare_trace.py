import json

import numpy as np
import pandas as pd
import pytest

from bimanual_collection.tools.compare_trace import compare_vectors, load_episode, load_trace, main


def write_trace(root):
    trace = root / "trace"
    trace.mkdir()
    (trace / "metadata.json").write_text(
        json.dumps({"task_description": "Pick one teabag."}),
        encoding="utf-8",
    )
    events = [
        {
            "event": "state_transition",
            "state": "RUNNING",
            "reason": "policy commands active",
        },
        {
            "event": "policy_observation",
            "observation_sequence": 1,
            "observation_timestamp_s": 10.0,
            "observation_state": list(range(12)),
            "cameras": [],
        },
        {
            "event": "policy_chunk",
            "observation_sequence": 1,
            "postprocessed_action_shape": [2, 12],
        },
        {
            "event": "robot_action",
            "source_timestamp_s": 10.0,
            "published_timestamp_s": 10.1,
            "action": list(range(20, 32)),
        },
    ]
    (trace / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return trace


def write_episode(root):
    episode = root / "episode-000001"
    episode.mkdir()
    (episode / "episode_metadata.json").write_text(
        json.dumps({"episode_id": "episode-000001", "task_description": "Pick one teabag."}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "timestep_index": 0,
                "monotonic_timestamp_s": 100.0,
                "left_follower_joints": np.arange(6, dtype=np.float32),
                "right_follower_joints": np.arange(6, 12, dtype=np.float32),
                "left_commanded_action": np.arange(20, 26, dtype=np.float32),
                "right_commanded_action": np.arange(26, 38, dtype=np.float32)[:6],
            }
        ]
    ).to_parquet(episode / "timesteps.parquet", index=False)
    return episode


def test_load_trace_groups_events(tmp_path):
    trace = load_trace(write_trace(tmp_path))

    assert trace.metadata["task_description"] == "Pick one teabag."
    assert len(trace.observations) == 1
    assert len(trace.robot_actions) == 1
    assert 1 in trace.chunks_by_sequence
    assert trace.state_transitions[0]["state"] == "RUNNING"


def test_compare_vectors_reports_state_and_action_metrics(tmp_path):
    trace = load_trace(write_trace(tmp_path))
    _metadata, episode_df = load_episode(write_episode(tmp_path))

    df = compare_vectors(trace, episode_df)

    row = df.iloc[0]
    assert row["state_mean_abs"] == 0.0
    assert row["left_action_mean_abs"] == 0.0
    assert bool(row["action_dim_match"])
    assert row["right_action_mean_abs"] == 0.0


def test_compare_trace_cli_writes_report_and_csv(tmp_path):
    trace = write_trace(tmp_path)
    episode = write_episode(tmp_path)
    output = tmp_path / "comparison"

    main(["--trace-dir", str(trace), "--episode-dir", str(episode), "--output-dir", str(output)])

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    vector_df = pd.read_csv(output / "vector_comparison.csv")
    assert report["task_match"] is True
    assert report["compared_steps"] == 1
    assert vector_df.loc[0, "action_mean_abs"] == 0.0
    assert (output / "index.html").exists()
    assert "Bimanual Trace Comparison" in (output / "index.html").read_text(encoding="utf-8")


def test_compare_trace_cli_rejects_missing_overlap(tmp_path):
    trace = write_trace(tmp_path)
    episode = write_episode(tmp_path)

    with pytest.raises(SystemExit):
        main([
            "--trace-dir",
            str(trace),
            "--episode-dir",
            str(episode),
            "--trace-start-index",
            "100",
        ])
