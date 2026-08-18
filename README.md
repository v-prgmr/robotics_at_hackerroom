# Orbit Bimanual Collection

Production-oriented bimanual SO-100 teleoperation and data recording for LeRobot `0.4.4`.

## Install

This repository is a `uv` project named `orbit` and pins Python to `3.10` because the LeRobot/Torch video stack is safer on Python `<3.13`.

```bash
uv sync --extra test
```

## Experimental LeWorldModel

The isolated LeWorldModel experiment, complete environment-variable reference, Orbit-to-Lance conversion procedure, smoke tests, full training launch, and held-out evaluation commands are documented in [`lewm_orbit_bridge/README.md`](lewm_orbit_bridge/README.md).

## List Hardware

```bash
uv run python bimanual_teleop.py --list-ports
uv run python bimanual_teleop.py --list-cameras
```

## Identify Ports And Cameras

Use the interactive finders to map physical devices to config keys.

```bash
uv run bimanual-find-ports --output config/local_ports.yaml
```

The serial-port wizard asks you to unplug exactly one arm at a time, compares the port list before and after unplugging, then writes:

```yaml
left_robot_port: /dev/serial/by-id/...
right_robot_port: /dev/serial/by-id/...
left_leader_port: /dev/serial/by-id/...
right_leader_port: /dev/serial/by-id/...
```

By default it prefers stable `/dev/serial/by-id` or `/dev/serial/by-path` aliases. Use `--prefer-raw` if you want `/dev/ttyACM*` paths.

For cameras:

```bash
uv run bimanual-find-cameras \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fps 30 \
    --output-dir outputs/camera_probe \
    --output config/local_cameras.yaml
```

The camera wizard probes OpenCV devices, saves one PNG per valid camera plus `outputs/camera_probe/contact_sheet.jpg`, asks you to assign `overhead`, `left_wrist`, and `right_wrist`, then writes:

```yaml
overhead_camera: /dev/v4l/by-id/...
left_wrist_camera: /dev/v4l/by-id/...
right_wrist_camera: /dev/v4l/by-id/...
camera_width: 1280
camera_height: 720
camera_fps: 30
```

By default it prefers stable `/dev/v4l/by-id` or `/dev/v4l/by-path` aliases. Use `--prefer-raw` if you want `/dev/video*` paths.

Before teleop or recording, validate the configured cameras:

```bash
uv run bimanual-teleop --config config/combined.yaml --check-cameras
```

This resolves every camera path, rejects duplicate physical devices, opens each camera, captures one frame, and prints the actual resolution/FPS. All configured cameras must resolve to different `/dev/video*` devices.

## Record

```bash
uv run python bimanual_teleop.py \
    --left-robot-port /dev/ttyACM0 \
    --right-robot-port /dev/ttyACM1 \
    --left-leader-port /dev/ttyACM2 \
    --right-leader-port /dev/ttyACM3 \
    --overhead-camera /dev/video0 \
    --left-wrist-camera /dev/video2 \
    --right-wrist-camera /dev/video4 \
    --robot-fps 60 \
    --camera-fps 30 \
    --camera-width 1280 \
    --camera-height 720 \
    --record \
    --task-description "Pick up the object bimanually" \
    --output-dir ./data/bimanual
```

You can also use `--config bimanual_collection/config/example.yaml` and override individual CLI arguments.

The current teleop CLI accepts one config file. If you generate separate `local_ports.yaml` and `local_cameras.yaml`, merge them into one local config file before recording, or pass the generated values as CLI overrides.

Press `q` for emergency stop. `Ctrl+C` triggers the same graceful shutdown path.

## Recording Controls

With `--record`, episodes are written under `output_dir`. The default is `./data/bimanual` relative to the repository, so the active local default is:

```text
/home/vrazer/workspace/orbit/data/bimanual
```

Manual episode control can be enabled in config:

```yaml
recording_control:
  manual_start: true
  start_save_key: r
  cancel_key: null
  status_interval_s: 5.0
```

In manual mode, teleop starts live but recording is idle until you press `r`:

- `r`: start recording when idle; save the current episode when recording.
- Keyboard cancel is disabled when `cancel_key: null`. This avoids collisions with foot switches that type `c` for pedal 3.
- Foot pedal `KEY_C` short press: start recording if idle, or save the current episode if recording. After saving, recording stays idle until the next third-pedal press.
- Foot pedal `KEY_C` hold: cancel/discard the current episode and stay idle when held for `recording_hold_cancel_s`.
- `q`: emergency stop; an active episode is saved on clean stop.

The control loop prints the dataset directory, hotkeys, operator-friendly episode number, save path, sample count, and periodic recording status. New episodes are saved as `episode-000001`, `episode-000002`, etc., and displayed as `Episode 1`, `Episode 2`, etc. Cancelled episodes are deleted from the temporary episode directory and are not published into the dataset.

To start a run at a specific episode number:

```bash
uv run bimanual-teleop --config config/combined.yaml --record --output-dir ./teabags_kitting_50_v1 --episode-start-number 51
```

This starts at `Episode 51` / `episode-000051`. If that directory already exists, the recorder uses the next available number.

CLI overrides are also available: `--recording-manual-start` / `--no-recording-manual-start`, `--record-start-save-key`, `--record-cancel-key`, and `--recording-status-interval-s`. Set `cancel_key: null` in YAML, or pass `--record-cancel-key ""`, to disable keyboard cancel.

Footswitch recording control example:

```yaml
footswitch:
  enabled: true
  backend: evdev
  third_pedal_mode: recording
  recording_hold_cancel_s: 1.0
  evdev:
    pause_code: KEY_C
```

The config key is still named `pause_code` because it identifies the third physical pedal; `third_pedal_mode: recording` changes that pedal from pause/resume to recording start/save/cancel.

## Teleop Startup Control

Teleop can move both followers to calibrated safe home poses before the live loop starts, then initialize control in one of two modes:

- `leader_absolute`: first live command targets the current leader pose, matching the old startup behavior where followers move to leaders.
- `hold_current`: first live command holds the current follower pose; follower motion starts only from subsequent leader deltas.

Example:

```yaml
teleop_control:
  startup_alignment: leader_absolute
  move_followers_to_home: true
  home_position_file: ./calibration/so100_bimanual/home_positions.yaml
  home_move_duration_s: 2.0
  home_move_steps: 120
```

CLI overrides are also available: `--startup-alignment`, `--move-followers-to-home` / `--no-move-followers-to-home`, `--home-position-file`, `--home-move-duration-s`, and `--home-move-steps`.

If `move_followers_to_home` is enabled, place the workspace in a clear state before starting teleop. For `leader_absolute`, place the leaders at the matching home pose before starting if you want zero initial motion after homing.

## Foot-Switch Clutch And Pause

The teleop loop supports a 3-button foot switch:

- Pedal 1: hold left-arm clutch
- Pedal 2: hold right-arm clutch
- Pedal 3: pause/resume data collection

Keyboard-style foot switch example:

```yaml
footswitch:
  enabled: true
  backend: keyboard
  debounce_s: 0.05
  keyboard:
    left_clutch_key: "1"
    right_clutch_key: "2"
    pause_key: "3"
```

Linux input-event example:

```yaml
footswitch:
  enabled: true
  backend: evdev
  debounce_s: 0.05
  evdev:
    device: /dev/input/by-id/usb-your-footswitch-event-kbd
    left_clutch_code: KEY_1
    right_clutch_code: KEY_2
    pause_code: KEY_3
```

While a clutch pedal is held, the corresponding follower arm holds its last commanded joint position, including gripper, and the corresponding leader can be repositioned freely. The other arm continues operating normally. On clutch release, the current leader and follower poses become the new relative-control references so the follower does not jump.

Pedal 3 toggles recording pause. While paused, both followers hold their last commanded positions, sensors continue being read, cameras continue being matched, and no new episode samples are written. On resume, both arms reset their leader/follower references before new motion commands are generated.

If `footswitch.third_pedal_mode: recording`, pedal 3 no longer toggles pause. Short-press it to start recording when idle or save the current episode when recording. After saving, the next episode does not start automatically; press pedal 3 again to start it. Hold pedal 3 to cancel/discard the current episode and stay idle.

Samples are skipped while either clutch is active or recording is paused. State transitions are saved in `control_events.parquet` for debugging.

## Calibrate Arms

Use `bimanual-calibrate` to run LeRobot's native SO-100 calibration flow for each arm role and save role-specific calibration JSON files. After each arm calibration, the tool also prompts you to place that arm in its safe home pose and captures it in `home_positions.yaml` unless `--skip-home-capture` is passed.

First ensure your config contains the four port keys. Add a calibration directory and stable role ids:

```yaml
calibration_dir: ./calibration/so100_bimanual
calibration_ids:
  left_follower: left_follower
  right_follower: right_follower
  left_leader: left_leader
  right_leader: right_leader
teleop_control:
  home_position_file: ./calibration/so100_bimanual/home_positions.yaml
```

Calibrate all four arms in sequence:

```bash
uv run bimanual-calibrate --config config/local_practice.yaml
```

Calibrate only one role:

```bash
uv run bimanual-calibrate \
    --config config/local_practice.yaml \
    --roles left_follower
```

The tool follows the same pattern as `lerobot-calibrate`: it constructs the actual LeRobot `SO100Follower` or `SO100Leader`, connects with `calibrate=False`, calls `device.calibrate()`, saves the calibration JSON, and disconnects. During normal practice or recording, `bimanual_teleop.py` uses the same `calibration_dir` and ids, so LeRobot loads the saved files automatically.

Expected files:

```text
calibration/so100_bimanual/
├── left_follower.json
├── right_follower.json
├── left_leader.json
├── right_leader.json
└── home_positions.yaml
```

## Live Inference Visualization

Policy trajectories are sampled independently from the robot command loop. The
robot loop remains at `robot_fps` (normally 60 Hz), while policy knots default to
16.57 Hz to match the measured spacing of the v2 training capture. Every robot
tick receives a monotonic-time linear interpolation between adjacent policy
knots. Inference requests latch and hold the request observation pose, and each
response uses a short bridge from the latest measured pose into action 0 before
trajectory playback begins. Configure this with `--policy-action-hz` and
`--action-bridge-duration-s`; `--chunk-execution-mode full` is the safe default.

`bimanual-inference` can stream runtime telemetry to Rerun while the policy is executing:

```bash
uv run bimanual-inference \
    --config config/combined.yaml \
    --checkpoint outputs/train/.../pretrained_model \
    --rerun-live
```

The live viewer logs the policy camera inputs, follower joint state, commanded actions, predicted action chunks, inference latency, action age, queue depth, camera staleness, holds, and deployment state transitions. Rerun logging is handled on a bounded background queue so visualization cannot block the robot loop; if the viewer falls behind, new telemetry events are dropped and the drop count is reported at shutdown.

To record the same telemetry for replay without opening a viewer:

```bash
uv run bimanual-inference \
    --config config/combined.yaml \
    --checkpoint outputs/train/.../pretrained_model \
    --rerun-save outputs/inference_rerun/run.rrd
```

Use `--rerun-connect-grpc` to stream to an already-running Rerun viewer, `--rerun-camera-fps` to throttle image logging, and `--rerun-max-queue` to tune the nonblocking event queue.

To replay saved debug-trace images from every `run-*` under a trace directory:

```bash
uv run bimanual-trace-rerun --trace-dir outputs/maniflow_inference_debug
```

To save that replay as a Rerun recording:

```bash
uv run bimanual-trace-rerun \
    --trace-dir outputs/maniflow_inference_debug \
    --rerun-save outputs/maniflow_inference_debug/images.rrd \
    --no-spawn
```

To export trace images directly to MP4 without opening Rerun:

```bash
uv run bimanual-trace-rerun \
    --trace-dir outputs/maniflow_inference_debug/run-... \
    --video-out outputs/maniflow_inference_debug/run.mp4 \
    --no-rerun
```

For every run under a parent trace directory, pass an output directory:

```bash
uv run bimanual-trace-rerun \
    --trace-dir outputs/maniflow_inference_debug \
    --video-out outputs/maniflow_inference_debug/videos \
    --no-rerun
```

The MP4 export tiles all cameras in each observation frame. Use `--camera overhead`, `--video-fps`, `--video-frame-limit`, `--video-tile-width`, and `--video-tile-height` to filter or resize the export.

## ManiFlow HIL Corrections

`bimanual-maniflow-hil` collects correction-only HIL data while a remote ManiFlow policy is running. Use this when the policy reaches a failure state and you want to recover with the leader arms, demonstrate the correct continuation, and save that intervention as a new training episode.

Start the ManiFlow policy server in the ManiFlow environment first, then run HIL from the Orbit environment:

```bash
uv run bimanual-maniflow-hil \
    --config config/combined.yaml \
    --maniflow-server http://127.0.0.1:8765 \
    --hil-protocol continuous \
    --hil-pedal-backend evdev \
    --hil-pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
    --hil-pause-code KEY_1 \
    --hil-correction-code KEY_2 \
    --hil-clutch-code KEY_3 \
    --hil-arm-code space \
    --hil-reset-code r \
    --output-dir ./dataset/maniflow_hil_corrections
```

The HIL controls are:

- `KEY_1`: pause/resume autonomous ManiFlow control.
- `KEY_2`: from paused, enter non-recording correction setup; from setup, start recording; from recording, stop/save the correction.
- `KEY_3`: hold clutch during setup or correction so both leaders can be recentered without moving the followers.
- `space`: arm/start autonomy after homing or reset.
- `r`: cancel/reset/home. If a correction is actively recording, its temporary local episode folder is discarded instead of saved.

The HIL phases are:

- `READY_TO_ARM`: followers are homed or held after reset; press `space` to start autonomy.
- `AUTONOMOUS`: ManiFlow publishes action chunks and controls the followers.
- `PAUSED`: queued ManiFlow actions are cleared, policy inference is reset, and the followers hold position.
- `PRE_CORRECTION`: leaders control the followers with the same relative `hold_current` teleop, but no frames are recorded. Use this to recenter leaders, align gripper inputs, or move to the exact correction start state.
- `CORRECTING`: the leaders control the followers with relative `hold_current` teleop and the correction frames are recorded.
- `ROLLOUT_TERMINATED`: the rollout must be reset before autonomy can begin again; press `r` to cancel/reset/home, physically reset the scene, then press `space` to start a fresh rollout.

Follower teleoperation remains at `robot_fps` (60 Hz), while correction rows are persisted at `policy_action_hz` (16.57 Hz for the tea-bag dataset) to match the expert demonstrations and autonomous rollout collections.

Supported intervention protocols are selected with `--hil-protocol`:

- `continuous` is the default and preserves the original behavior. Multiple correction windows can be collected during one physical rollout, and every correction window is saved as a separate episode.
- `rac` enforces strict recovery-and-correction collection. One saved correction terminates the rollout; autonomous control cannot resume until the operator presses `r` to reset/home and then `space` to arm a fresh rollout.
- `bounded` allows at most `--hil-max-interventions-per-rollout` saved corrections before terminating the rollout. The default limit is `2`; a limit of `1` behaves like `rac`.

For teabag-kitting downstream-failure discovery, bounded mode with two interventions is usually the recommended compromise:

```bash
uv run bimanual-maniflow-hil \
    --config config/combined.yaml \
    --maniflow-server http://127.0.0.1:8765 \
    --hil-protocol bounded \
    --hil-max-interventions-per-rollout 2 \
    --hil-pedal-backend evdev \
    --hil-pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
    --hil-pause-code KEY_1 \
    --hil-correction-code KEY_2 \
    --hil-clutch-code KEY_3 \
    --hil-arm-code space \
    --hil-reset-code r \
    --output-dir ./dataset/maniflow_hil_bounded
```

For strict RaC collection, use one correction per physical rollout:

```bash
uv run bimanual-maniflow-hil \
    --config config/combined.yaml \
    --maniflow-server http://127.0.0.1:8765 \
    --hil-protocol rac \
    --hil-pedal-backend evdev \
    --hil-pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd \
    --hil-pause-code KEY_1 \
    --hil-correction-code KEY_2 \
    --hil-clutch-code KEY_3 \
    --hil-arm-code space \
    --hil-reset-code r \
    --output-dir ./dataset/maniflow_hil_rac
```

Leader arms do not need to match the failed follower pose before takeover. When pre-correction setup starts, the current leader pose and current follower pose become the relative-control references, so leader motion is applied as a delta from wherever the leaders are resting. Hold the clutch pedal to recenter the leaders or align gripper inputs while the followers stay fixed; release it to continue from new references with no follower jump.

The recommended correction workflow is:

- After startup homing, press `space` to arm autonomy and start the rollout.
- Press `KEY_1` when failure is imminent or has just occurred. This pauses autonomy and holds the followers.
- Press `KEY_2` once to enter `PRE_CORRECTION`. Leaders now have control, but nothing is recorded.
- Use `KEY_3` clutch as needed to recenter leaders or match leader gripper state to the follower state without moving the followers.
- Optionally use leader control in `PRE_CORRECTION` to move the robot to the exact state where the useful recovery demonstration should begin.
- Press `KEY_2` again to enter `CORRECTING` and start recording.
- Demonstrate recovery, correct retry, and successful completion of the current subtask.
- Press `KEY_2` again to stop and save the correction episode.
- Press `r` at any time to cancel/reset/home. If a correction was recording, it is discarded and not counted. The harness returns to `READY_TO_ARM`; physically reset the scene, then press `space` to start the next rollout.

Only `CORRECTING` windows are recorded. `PRE_CORRECTION` setup, autonomous ManiFlow actions, paused holds, policy action chunks, reset motion, and clutch-only recentering are intentionally not written as training labels. Empty correction windows are discarded and not counted. Each saved correction window is saved as an intermediate-format episode under `--output-dir`.

When you resume autonomous control, the HIL loop clears the local action queue, clears the observation buffer, increments a policy generation id, and calls the ManiFlow runtime reset. Delayed policy responses from older generations are discarded. ManiFlow then receives fresh images and follower joint state from the corrected physical scene, so the next action chunk is conditioned on the corrected state rather than stale pre-intervention history.

Every saved correction episode includes HIL metadata such as `collection_type=hil`, `hil_protocol`, `contains_autonomous_actions=false`, `intervention_index`, `rollout_intervention_count`, `prior_human_intervention`, `rollout_terminated_after_intervention`, `policy_name=maniflow`, `policy_server`, `rollout_id`, and `policy_generation_id`. Bounded episodes also include `max_interventions_per_rollout`. Optional analysis fields `failure_reason`, `subtask`, and `correction_success` are written as `null` placeholders so collection requires no live typing and can be annotated later if useful.

After collecting corrections, export them to LeRobot and convert to ManiFlow zarr:

To inspect the raw intermediate correction dataset before export, replay it directly in Rerun:

```bash
uv run bimanual-dataset-rerun \
    --dataset ./dataset/maniflow_hil_bounded
```

For sequential review without loading every episode at once, use lazy mode. This opens a small companion window with `Previous`, `Reload`, and `Next` buttons; only the selected episode is decoded and pushed to Rerun. Moving to another episode clears the previously logged timestep entities and reuses the same active recording, so old episodes should not accumulate on the timeline:

```bash
uv run bimanual-dataset-rerun \
    --dataset ./dataset/maniflow_hil_bounded \
    --lazy
```

Keyboard shortcuts in the companion window are left/right arrows for previous/next, `r` to reload, and `q` to close the navigator.

Replay one episode, save an `.rrd`, or stream to an existing viewer:

```bash
uv run bimanual-dataset-rerun \
    --dataset ./dataset/maniflow_hil_bounded \
    --episode episode-000001 \
    --frame-stride 2 \
    --rerun-save ./dataset/maniflow_hil_bounded/episode-000001.rrd

uv run bimanual-dataset-rerun \
    --dataset ./dataset/maniflow_hil_bounded \
    --rerun-connect-grpc rerun+http://127.0.0.1:9876/proxy \
    --no-spawn
```

`bimanual-dataset-rerun` reads `timesteps.parquet`, `episode_metadata.json`, and `videos/*.mp4` directly. It does not require LeRobot export. It logs camera frames, leader and follower joint scalars, commanded correction actions, and episode metadata.

Export to LeRobot and convert to ManiFlow zarr when you are ready to train:

```bash
uv run bimanual-export-lerobot \
    --input-dir ./dataset/maniflow_hil_corrections \
    --output-dir ./dataset/maniflow_hil_corrections_lerobot \
    --repo-id local/maniflow_hil_corrections \
    --video-codec h264 \
    --encoder-threads 4

python maniflow_orbit_bridge/convert_orbit_lerobot_to_maniflow.py \
    --lerobot-root ./dataset/maniflow_hil_corrections_lerobot \
    --output-zarr ./dataset/maniflow_hil_corrections_maniflow.zarr \
    --image-size 224 \
    --overwrite
```

## Data Layout

The default backend writes a clean intermediate format:

```text
data/bimanual/
├── dataset_metadata.json
└── episode-*/
    ├── episode_metadata.json
    ├── timesteps.parquet
    ├── camera_index.parquet
    ├── control_events.parquet
    └── videos/
        ├── overhead.mp4
        ├── left_wrist.mp4
        └── right_wrist.mp4
```

`timesteps.parquet` contains robot state, leader state, commanded actions, timing diagnostics, and camera references. `camera_index.parquet` maps every robot timestep to camera video frame indices and preserves original camera timestamps, frame age, stale flags, missing flags, dropped-frame counts, and disconnect state.

`control_events.parquet` records clutch and pause transitions, including `left_clutch_active`, `right_clutch_active`, and `recording_paused` for debugging cycles that were intentionally not written as training samples.

## Backfill Task Instructions

For existing intermediate recordings, update the task/instruction stored in each `episode_metadata.json` before exporting to LeRobot:

```bash
uv run bimanual-set-task \
    --dataset ./dataset/teabags_kitting_50_v1 \
    --task "Pick one teabag and place it into the kitting tray."
```

Update only selected episodes with numbers, ranges, or full episode ids:

```bash
uv run bimanual-set-task \
    --dataset ./dataset/teabags_kitting_50_v1 \
    --task "Pick one teabag and place it into the kitting tray." \
    --episodes 1 2 7-10 episode-000012
```

Use `--dry-run` to preview changes. Re-export with `bimanual-export-lerobot --overwrite` afterward because LeRobot stores tasks in its own metadata and frame data.

## Design Decisions

The robot loop never waits for cameras. Each camera has an asynchronous capture thread and a bounded ring buffer. At each 60 Hz robot timestep, the loop assigns one shared monotonic timestamp and performs nearest-frame matching against each camera buffer.

Videos store each unique matched camera frame once. If a 60 Hz robot loop maps multiple timesteps to the same 30 Hz camera frame, the timestamp index points multiple timesteps to the same video frame. This keeps video efficient while retaining exact timestep-to-frame correspondence.

The recorder is independent of hardware. It accepts synchronized `TimestepSample` objects and writes episodes atomically through a temporary directory rename, so interrupted recordings do not corrupt completed episodes.

## Validate

```bash
uv run bimanual-validate ./data/bimanual
```

The validator checks timestamp monotonicity, missing robot states, missing camera frames, stale frames, jitter, episode length consistency, video references, and left/right state/action dimensions.

## LeRobot Export

The intermediate format can be deterministically converted to LeRobot v3.0 using `bimanual-export-lerobot`. The converter duplicates indexed camera frames as needed to produce one LeRobot observation per robot timestep.

```bash
uv run bimanual-export-lerobot \
    --input-dir ./teabags_kitting_50_v1 \
    --output-dir ./teabags_kitting_50_v1_lerobot \
    --repo-id vrazer/teabags_kitting_50_v1 \
    --video-codec h264 \
    --encoder-threads 1
```

`--output-dir` must not already exist unless `--overwrite` is passed.

By default the exporter measures the source `monotonic_timestamp_s` cadence and uses the nearest integer FPS supported by LeRobot. It rejects an explicit `--fps` that differs from the measured row rate by more than 10%. Use `--allow-fps-mismatch` only for an intentional temporal resampling workflow.

The exporter streams camera frames from MP4s instead of caching full videos in memory. `--video-codec h264 --encoder-threads 1` is the lower-memory default. `libsvtav1` is supported by LeRobot but can use much more RAM on large multi-camera exports.

View the exported dataset with LeRobot's Rerun viewer:

```bash
uv run lerobot-dataset-viz \
    --repo-id vrazer/teabags_kitting_50_v1 \
    --root ./teabags_kitting_50_v1_lerobot \
    --episode-index 0
```

Episode indices in `lerobot-dataset-viz` are zero-based, so Orbit `Episode 1` is `--episode-index 0`.

## Test Plan

```bash
uv run pytest
```

Basic tests cover nearest-frame matching, stale/missing frame metadata, atomic episode save, and discard behavior.
