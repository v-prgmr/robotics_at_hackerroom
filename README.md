# Orbit Bimanual Collection

Production-oriented bimanual SO-100 teleoperation and data recording for LeRobot `0.4.4`.

## Install

This repository is a `uv` project named `orbit` and pins Python to `3.10` because the LeRobot/Torch video stack is safer on Python `<3.13`.

```bash
uv sync --extra test
```

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

The intermediate format can be deterministically converted to LeRobot v3.0 using `bimanual_collection.recording.backends.lerobot_export.export_to_lerobot`. The converter duplicates indexed camera frames as needed to produce one LeRobot observation per robot timestep.

## Test Plan

```bash
uv run pytest
```

Basic tests cover nearest-frame matching, stale/missing frame metadata, atomic episode save, and discard behavior.
