import json

import pytest

from bimanual_collection.tools.set_task import parse_episode_selector, set_task_description


def make_episode(dataset_dir, name, task=""):
    episode = dataset_dir / name
    episode.mkdir(parents=True)
    metadata = {
        "episode_id": name,
        "task_description": task,
        "operator_notes": "keep me",
    }
    (episode / "episode_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return episode


def read_task(episode):
    return json.loads((episode / "episode_metadata.json").read_text(encoding="utf-8"))["task_description"]


def test_parse_episode_selector_accepts_numbers_ranges_and_names():
    selected = parse_episode_selector(["1", "000002", "4-5", "episode-000009"])

    assert selected == {
        "episode-000001",
        "episode-000002",
        "episode-000004",
        "episode-000005",
        "episode-000009",
    }


def test_set_task_description_updates_all_episode_metadata(tmp_path):
    first = make_episode(tmp_path, "episode-000001", task="old")
    second = make_episode(tmp_path, "episode-000002", task="")

    updated = set_task_description(tmp_path, "Pick a teabag")

    assert [path.parent.name for path in updated] == ["episode-000001", "episode-000002"]
    assert read_task(first) == "Pick a teabag"
    assert read_task(second) == "Pick a teabag"
    metadata = json.loads((first / "episode_metadata.json").read_text(encoding="utf-8"))
    assert metadata["operator_notes"] == "keep me"


def test_set_task_description_updates_selected_episodes_only(tmp_path):
    first = make_episode(tmp_path, "episode-000001", task="old 1")
    second = make_episode(tmp_path, "episode-000002", task="old 2")

    set_task_description(tmp_path, "Selected task", selected={"episode-000002"})

    assert read_task(first) == "old 1"
    assert read_task(second) == "Selected task"


def test_set_task_description_dry_run_does_not_write(tmp_path):
    episode = make_episode(tmp_path, "episode-000001", task="old")

    set_task_description(tmp_path, "New task", dry_run=True)

    assert read_task(episode) == "old"


def test_set_task_description_errors_for_missing_selected_episode(tmp_path):
    make_episode(tmp_path, "episode-000001")

    with pytest.raises(FileNotFoundError):
        set_task_description(tmp_path, "Task", selected={"episode-000002"})
