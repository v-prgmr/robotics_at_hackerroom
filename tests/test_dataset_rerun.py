import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bimanual_collection.tools import dataset_rerun


class FakeRerun:
    class Image:
        def __init__(self, image):
            self.image = image

    class Scalars:
        def __init__(self, value):
            self.value = value

    class TextDocument:
        def __init__(self, text, media_type=None):
            self.text = text
            self.media_type = media_type

    class Clear:
        def __init__(self, recursive=False):
            self.recursive = recursive

    def __init__(self):
        self.inits = []
        self.saves = []
        self.times = []
        self.logs = []

    def init(self, application_id, spawn=True):
        self.inits.append((application_id, spawn))

    def connect_grpc(self, url):
        self.grpc_url = url

    def save(self, path):
        self.saves.append(path)

    def set_time_sequence(self, name, value):
        self.times.append(("sequence", name, value))

    def set_time_seconds(self, name, value):
        self.times.append(("seconds", name, value))

    def disable_timeline(self, name):
        self.times.append(("disabled", name, None))

    def log(self, entity, value):
        self.logs.append((entity, value))


def write_video(path: Path, values: list[int]) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (8, 6))
    assert writer.isOpened()
    try:
        for value in values:
            frame = np.full((6, 8, 3), value, dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def make_episode(root: Path, name: str = "episode-000001") -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "episode_id": [name, name],
            "timestep_index": [0, 1],
            "monotonic_timestamp_s": [10.0, 10.1],
            "wall_timestamp_s": [100.0, 100.1],
            "left_follower_joint_names": [["shoulder.pos"], ["shoulder.pos"]],
            "right_follower_joint_names": [["shoulder.pos"], ["shoulder.pos"]],
            "left_leader_joint_names": [["shoulder.pos"], ["shoulder.pos"]],
            "right_leader_joint_names": [["shoulder.pos"], ["shoulder.pos"]],
            "left_action_names": [["shoulder.pos"], ["shoulder.pos"]],
            "right_action_names": [["shoulder.pos"], ["shoulder.pos"]],
            "left_follower_joints": [[1.0], [1.1]],
            "right_follower_joints": [[2.0], [2.1]],
            "left_leader_joints": [[3.0], [3.1]],
            "right_leader_joints": [[4.0], [4.1]],
            "left_commanded_action": [[5.0], [5.1]],
            "right_commanded_action": [[6.0], [6.1]],
            "overhead_video_frame_index": [0, 1],
        }
    )
    df.to_parquet(episode / "timesteps.parquet", index=False)
    (episode / "episode_metadata.json").write_text(
        json.dumps({"episode_id": name, "task_description": "test", "hil_protocol": "bounded"}),
        encoding="utf-8",
    )
    write_video(episode / "videos" / "overhead.mp4", [10, 20])
    return episode


def make_topreward_results(root: Path, *, dataset_name: str, episode_id: str = "episode-000001") -> Path:
    output = root / "topreward"
    output.mkdir()
    pd.DataFrame(
        {
            "dataset": [dataset_name, dataset_name, "another_dataset"],
            "episode_id": [episode_id, episode_id, episode_id],
            "timestep_index": [0, 1, 0],
            "is_anchor": [True, False, True],
            "topreward_logp_true": [-3.0, -1.0, -10.0],
            "topreward_logp_false": [-1.0, -2.0, -1.0],
            "topreward_true_false_margin": [-2.0, 1.0, -9.0],
        }
    ).to_parquet(output / "timestep_scores.parquet", index=False)
    pd.DataFrame(
        {
            "dataset": [dataset_name],
            "episode_id": [episode_id],
            "collection_type": ["expert"],
            "terminal_logp_true": [-1.0],
            "terminal_true_false_margin": [1.0],
            "success_score": [np.nan],
        }
    ).to_parquet(output / "episode_scores.parquet", index=False)
    return output


def test_discover_episode_dirs_supports_numbers_and_names(tmp_path):
    first = make_episode(tmp_path, "episode-000001")
    second = make_episode(tmp_path, "episode-000002")
    (tmp_path / ".episode-000003.tmp-1").mkdir()

    assert dataset_rerun.discover_episode_dirs(tmp_path) == [first, second]
    assert dataset_rerun.discover_episode_dirs(tmp_path, episodes=["2"]) == [second]
    assert dataset_rerun.discover_episode_dirs(tmp_path, episodes=["episode-000001"]) == [first]


def test_replay_episode_logs_images_and_scalars(tmp_path):
    episode = make_episode(tmp_path)
    rr = FakeRerun()

    count = dataset_rerun.replay_episode(rr, episode, episode_index=3, cameras={"overhead"})

    assert count == 2
    image_logs = [entry for entry in rr.logs if entry[0] == "cameras/overhead/image"]
    assert len(image_logs) == 2
    assert image_logs[0][1].image.shape == (6, 8, 3)
    assert ("sequence", "episode", 3) in rr.times
    assert ("sequence", "timestep", 1) in rr.times
    assert any(entity == "followers/left/shoulder_pos" and value.value == pytest.approx(1.0) for entity, value in rr.logs)


def test_replay_episode_logs_topreward_curves_and_anchors(tmp_path):
    dataset = tmp_path / "teabags_kitting_50_v2"
    episode = make_episode(dataset)
    score_dir = make_topreward_results(tmp_path, dataset_name=dataset.name)
    results = dataset_rerun.load_topreward_results(score_dir, dataset_name=dataset.name)
    rr = FakeRerun()

    dataset_rerun.replay_episode(rr, episode, episode_index=0, topreward_results=results)

    values = {entity: value.value for entity, value in rr.logs if hasattr(value, "value")}
    assert values["topreward/interpolated/logp_true"] == pytest.approx(-1.0)
    assert values["topreward/interpolated/true_false_margin"] == pytest.approx(1.0)
    assert values["topreward/within_episode_normalized/logp_true"] == pytest.approx(1.0)
    anchor_logs = [entity for entity, _value in rr.logs if entity.startswith("topreward/measured_anchors/")]
    assert len(anchor_logs) == 3


def test_load_topreward_results_filters_dataset_and_rejects_duplicates(tmp_path):
    score_dir = make_topreward_results(tmp_path, dataset_name="selected")

    results = dataset_rerun.load_topreward_results(score_dir, dataset_name="selected")

    assert set(results.timestep_by_episode) == {"episode-000001"}
    scores = results.timestep_by_episode["episode-000001"]
    assert scores["topreward_true_normalized_plot"].tolist() == [0.0, 1.0]

    duplicate = pd.read_parquet(score_dir / "timestep_scores.parquet")
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(score_dir / "timestep_scores.parquet", index=False)
    with pytest.raises(ValueError, match="Duplicate TOPReward"):
        dataset_rerun.load_topreward_results(score_dir, dataset_name="selected")


def test_main_initializes_rerun_and_replays_dataset(tmp_path):
    make_episode(tmp_path)
    rr = FakeRerun()

    dataset_rerun.main(["--dataset", str(tmp_path), "--no-spawn", "--frame-limit", "1"], rr_module=rr)

    assert rr.inits == [("orbit_intermediate_dataset", False)]
    assert any(entity == "cameras/overhead/image" for entity, _value in rr.logs)


def test_main_accepts_topreward_output_directory(tmp_path):
    dataset = tmp_path / "teabags_kitting_50_v2"
    make_episode(dataset)
    score_dir = make_topreward_results(tmp_path, dataset_name=dataset.name)
    rr = FakeRerun()

    dataset_rerun.main(
        ["--dataset", str(dataset), "--topreward-dir", str(score_dir), "--no-spawn"],
        rr_module=rr,
    )

    assert any(entity == "topreward/interpolated/logp_true" for entity, _value in rr.logs)


def test_clear_rerun_recording_uses_clear_when_available():
    rr = FakeRerun()

    dataset_rerun.clear_rerun_recording(rr)

    cleared = {entity for entity, _value in rr.logs}
    assert {"cameras", "followers", "leaders", "actions", "topreward", "episode_metadata"}.issubset(cleared)
    assert all(value.recursive is True for _entity, value in rr.logs)


def test_clear_rerun_timesteps_clears_each_previous_timestep():
    rr = FakeRerun()

    dataset_rerun.clear_rerun_timesteps(rr, {0, 2})

    timestep_times = [entry for entry in rr.times if entry[1] == "timestep"]
    assert ("sequence", "timestep", 0) in timestep_times
    assert ("sequence", "timestep", 2) in timestep_times
    assert any(entity == "cameras" for entity, _value in rr.logs)
    assert ("disabled", "episode_time", None) in rr.times
    assert ("disabled", "monotonic_time", None) in rr.times
    assert ("disabled", "wall_time", None) in rr.times


def test_episode_summary_includes_hil_metadata(tmp_path):
    episode = make_episode(tmp_path)

    summary = dataset_rerun.episode_summary(episode, index=0, total=3)

    assert "Episode 1 of 3" in summary
    assert "episode-000001" in summary
    assert "Protocol: bounded" in summary
    assert "Task: test" in summary


def test_episode_summary_includes_topreward_metadata(tmp_path):
    dataset = tmp_path / "teabags_kitting_50_v2"
    episode = make_episode(dataset)
    score_dir = make_topreward_results(tmp_path, dataset_name=dataset.name)
    results = dataset_rerun.load_topreward_results(score_dir, dataset_name=dataset.name)

    summary = dataset_rerun.episode_summary(
        episode,
        index=0,
        total=1,
        topreward_results=results,
    )

    assert "TOPReward collection: expert" in summary
    assert "Terminal log P(True): -1.0000" in summary
    assert "Terminal True/False margin: 1.0000" in summary


def test_replayed_timesteps_matches_stride_and_limit(tmp_path):
    episode = make_episode(tmp_path)

    assert dataset_rerun.replayed_timesteps(episode) == {0, 1}
    assert dataset_rerun.replayed_timesteps(episode, frame_stride=2) == {0}
    assert dataset_rerun.replayed_timesteps(episode, frame_limit=1) == {0}
