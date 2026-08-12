import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = ROOT / "maniflow_orbit_bridge/vast_setup_maniflow_env.sh"
TRAIN_SCRIPT = ROOT / "maniflow_orbit_bridge/vast_train_maniflow_orbit.sh"
PROGRESS_TRAIN_SCRIPT = ROOT / "maniflow_orbit_bridge/runpod_train_maniflow_progress.sh"
VAST_PROGRESS_TRAIN_SCRIPT = ROOT / "maniflow_orbit_bridge/vast_train_maniflow_progress.sh"


def _write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/usr/bin/env bash\nset -u\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_train(tmp_path: Path, *, train_exit=0, destroy=True, success_only=True, helper_exit=0):
    train_log = tmp_path / "train.log"
    destroy_log = tmp_path / "destroy.log"
    trainer = _write_script(
        tmp_path / "trainer.sh",
        f'printf "%s\\n" "$*" > "{train_log}"\nexit {train_exit}',
    )
    destroy_helper = _write_script(
        tmp_path / "destroy.sh",
        f'printf "%s\\n" "$*" > "{destroy_log}"\nexit {helper_exit}',
    )
    env = {
        **os.environ,
        "WORKSPACE_DIR": str(tmp_path / "data"),
        "TRAIN_LAUNCHER": str(trainer),
        "VAST_DESTROY_HELPER": str(destroy_helper),
        "VAST_DESTROY_ON_EXIT": str(destroy).lower(),
        "VAST_DESTROY_ON_SUCCESS_ONLY": str(success_only).lower(),
        "VAST_API_KEY": "secret",
        "VAST_INSTANCE_ID": "4242",
    }
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "optimizer.lr=1e-5"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, train_log, destroy_log


def test_vast_wrappers_default_to_workspace_volume():
    expected = 'WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"'

    assert expected in SETUP_SCRIPT.read_text()
    assert expected in TRAIN_SCRIPT.read_text()
    assert expected in VAST_PROGRESS_TRAIN_SCRIPT.read_text()


def test_progress_launcher_maps_exact_gradient_steps_to_hydra():
    shared = PROGRESS_TRAIN_SCRIPT.read_text()
    vast = VAST_PROGRESS_TRAIN_SCRIPT.read_text()

    assert 'NUM_GRAD_STEPS="${NUM_GRAD_STEPS:-}"' in shared
    assert 'HYDRA_OVERRIDES+=("training.num_grad_steps=${NUM_GRAD_STEPS}")' in shared
    assert 'GRADIENT_ACCUMULATE_EVERY="${GRADIENT_ACCUMULATE_EVERY:-1}"' in shared
    assert '"training.gradient_accumulate_every=${GRADIENT_ACCUMULATE_EVERY}"' in shared
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"' in shared
    assert 'realpath --canonicalize-missing --no-symlinks "${SOURCE_CHECKPOINT}"' in shared
    assert '"source_checkpoint=\'${SOURCE_CHECKPOINT}\'"' in shared
    assert 'bash "${TRAIN_LAUNCHER}" "$@"' in vast


def test_vast_setup_wrapper_uses_persistent_volume_defaults(tmp_path):
    env_log = tmp_path / "env.log"
    setup = _write_script(
        tmp_path / "setup.sh",
        f'printf "%s|%s|%s|%s" "$WORKSPACE_DIR" "$MANIFLOW_DIR" "$CONDA_ENV_DIR" "$HF_HOME" > "{env_log}"',
    )
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT)],
        env={**os.environ, "WORKSPACE_DIR": str(tmp_path / "data"), "SETUP_LAUNCHER": str(setup)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    root = tmp_path / "data"
    assert env_log.read_text() == f"{root}|{root}/maniflow|{root}/conda_envs/maniflow|{root}/huggingface"


def test_vast_launcher_destroys_after_success(tmp_path):
    result, train_log, destroy_log = _run_train(tmp_path)

    assert result.returncode == 0, result.stderr
    assert train_log.read_text().strip() == "optimizer.lr=1e-5"
    assert destroy_log.read_text().strip() == "4242 secret"


def test_vast_launcher_preserves_instance_after_failure_by_default(tmp_path):
    result, _, destroy_log = _run_train(tmp_path, train_exit=7)

    assert result.returncode == 7
    assert not destroy_log.exists()
    assert "Not destroying Vast instance" in result.stdout


def test_vast_launcher_can_destroy_after_failure_when_explicit(tmp_path):
    result, _, destroy_log = _run_train(tmp_path, train_exit=7, success_only=False)

    assert result.returncode == 7
    assert destroy_log.read_text().strip() == "4242 secret"


def test_vast_launcher_reports_destroy_failure(tmp_path):
    result, _, _ = _run_train(tmp_path, helper_exit=9)

    assert result.returncode == 1
    assert "destruction failed" in result.stdout


def test_vast_launcher_requires_destroy_credentials_before_training(tmp_path):
    trainer = _write_script(tmp_path / "trainer.sh", "exit 0")
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env={
            **os.environ,
            "WORKSPACE_DIR": str(tmp_path / "data"),
            "TRAIN_LAUNCHER": str(trainer),
            "VAST_DESTROY_ON_EXIT": "true",
            "VAST_API_KEY": "",
            "VAST_INSTANCE_ID": "",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires VAST_API_KEY and VAST_INSTANCE_ID" in result.stdout
