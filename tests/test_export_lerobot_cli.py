from pathlib import Path

import pandas as pd
import pytest

from bimanual_collection.tools.export_lerobot import intermediate_episodes, main


def make_episode(root: Path, name: str = "episode-000001") -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    pd.DataFrame({"monotonic_timestamp_s": [0.0, 1.0 / 60.0, 2.0 / 60.0]}).to_parquet(
        episode / "timesteps.parquet", index=False
    )
    return episode


def test_intermediate_episodes_filters_temp_and_non_episode_dirs(tmp_path):
    episode = make_episode(tmp_path)
    (tmp_path / ".episode-000002.tmp-1").mkdir()
    (tmp_path / "notes").mkdir()

    assert intermediate_episodes(tmp_path) == [episode]


def test_export_cli_calls_exporter_with_expected_arguments(tmp_path):
    input_dir = tmp_path / "intermediate"
    output_dir = tmp_path / "lerobot"
    make_episode(input_dir)
    calls = []

    def exporter(intermediate_root, output_root, repo_id, fps, **kwargs):
        calls.append((intermediate_root, output_root, repo_id, fps, kwargs))
        output_root.mkdir()

    main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--repo-id",
            "vrazer/test_dataset",
            "--fps",
            "60",
        ],
        exporter=exporter,
    )

    assert calls == [(input_dir, output_dir, "vrazer/test_dataset", 60, {"vcodec": "h264", "encoder_threads": 1})]


def test_export_cli_infers_nearest_integer_source_fps(tmp_path):
    input_dir = tmp_path / "intermediate"
    output_dir = tmp_path / "lerobot"
    episode = make_episode(input_dir)
    pd.DataFrame({"monotonic_timestamp_s": [0.0, 1.0 / 16.57, 2.0 / 16.57]}).to_parquet(
        episode / "timesteps.parquet", index=False
    )
    calls = []

    def exporter(_intermediate_root, output_root, _repo_id, fps, **_kwargs):
        calls.append(fps)
        output_root.mkdir()

    main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--repo-id",
            "vrazer/test_dataset",
        ],
        exporter=exporter,
    )

    assert calls == [17]


def test_export_cli_rejects_large_explicit_fps_mismatch(tmp_path):
    input_dir = tmp_path / "intermediate"
    output_dir = tmp_path / "lerobot"
    episode = make_episode(input_dir)
    pd.DataFrame({"monotonic_timestamp_s": [0.0, 1.0 / 16.57, 2.0 / 16.57]}).to_parquet(
        episode / "timesteps.parquet", index=False
    )

    with pytest.raises(SystemExit):
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--repo-id",
                "vrazer/test_dataset",
                "--fps",
                "60",
            ],
            exporter=lambda *_args, **_kwargs: None,
        )


def test_export_cli_refuses_existing_output_without_overwrite(tmp_path):
    input_dir = tmp_path / "intermediate"
    output_dir = tmp_path / "lerobot"
    make_episode(input_dir)
    output_dir.mkdir()

    with pytest.raises(SystemExit):
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--repo-id",
                "vrazer/test_dataset",
            ],
            exporter=lambda *_args, **_kwargs: None,
        )


def test_export_cli_overwrite_removes_existing_output(tmp_path):
    input_dir = tmp_path / "intermediate"
    output_dir = tmp_path / "lerobot"
    make_episode(input_dir)
    output_dir.mkdir()
    stale = output_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    def exporter(_intermediate_root, output_root, _repo_id, _fps, **_kwargs):
        assert not stale.exists()
        output_root.mkdir()

    main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--repo-id",
            "vrazer/test_dataset",
            "--overwrite",
        ],
        exporter=exporter,
    )

    assert output_dir.exists()
