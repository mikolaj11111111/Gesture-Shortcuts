# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gesture Shortcuts is a Windows application that detects hand gestures using MediaPipe and YOLO classification to trigger keyboard shortcuts. The primary use cases are:
- Microphone control during video calls (mute/unmute)
- Nail-biting detection with notifications
- Telemetry logging to InfluxDB for behavior tracking

## Architecture

### Detection Pipeline

1. **MediaPipe Hand Landmarker** (`live_detection/detector_yolo.py:60-73`)
   - Detects hands and extracts 21 landmarks per hand
   - Calculates bounding box around detected hand
   - Runs in VIDEO mode for real-time performance

2. **Hand Cropping** (`live_detection/detector_yolo.py:109-120`)
   - Crops hand region with 20px margin
   - Resizes to YOLO input size (224x224)

3. **YOLO Classification** (`live_detection/detector_yolo.py:124-131`)
   - YOLOv8-nano classifier trained on hand gesture images
   - Categories: `continue_recording_sign`, `stop_recording_sign`, `nail_biting_sign`, `other`
   - Returns top-1 prediction with confidence score

4. **Action Triggering** (`actions/actions.py`)
   - Debouncing mechanism (cooldown per gesture)
   - Minimum confidence threshold (default 0.75)
   - Special handling for Teams integration (checks if Teams is running)
   - Threading.Timer for delayed actions (Win+H has 2s delay)

5. **Telemetry** (`telemetry/telemetry.py`)
   - Asynchronous batch writing to InfluxDB
   - Graceful degradation if database unavailable
   - Tracks: detections, activity durations, session stats

### Module Organization

```
├── main.py                          # Entry point with CLI args
├── live_detection/
│   ├── detector_yolo.py            # Main detection loop (YOLO-based)
│   └── detector.py                 # Alternative MediaPipe-only version
├── actions/
│   └── actions.py                  # Keyboard shortcuts & notifications
├── telemetry/
│   └── telemetry.py                # InfluxDB client (singleton)
├── creating_dataset/
│   ├── signs_recording.py          # Manual dataset recording tool
│   └── yolo_signs_recording.py     # YOLO-based recording tool
├── yolo_training/
│   └── train_yolo.py               # YOLOv8 classification training
└── models/
    └── hand_landmarker.task        # MediaPipe model (download separately)
└── venv/                           # Python virtual environment
```

## Common Commands

### Running the Application

```bash
# Standard mode (with actions and telemetry)
python main.py

# Disable keyboard actions (detection only)
python main.py --no-actions

# Disable telemetry logging
python main.py --no-telemetry

# Adjust confidence threshold
python main.py --confidence 0.8

# Select different camera
python main.py --camera 1
```

### Dataset Creation

```bash
# Record gesture samples using MediaPipe
# Press 'R' to start/stop recording, 'S' to save single frame, 'Q' to exit
python creating_dataset/signs_recording.py

# Record using YOLO detection
python creating_dataset/yolo_signs_recording.py
```

Dataset structure: `creating_dataset/data/{gesture_name}/images/*.jpg`

### Training YOLO Model

```bash
# Train YOLOv8 classifier on collected gestures
# Uses 80/20 train/val split, 50 epochs, early stopping
python yolo_training/train_yolo.py
```

Output: `yolo_training/runs/gesture_classification/weights/best.pt`

### Building Executable

```bash
# Build standalone .exe with PyInstaller
build.bat

# Or manually:
pyinstaller gesture_shortcuts.spec
```

Output: `dist/GestureShortcuts/GestureShortcuts.exe`

### Telemetry Setup

```bash
# Start InfluxDB + Grafana via Docker
docker-compose up -d

# Verify InfluxDB health
curl http://localhost:8086/health

# Stop containers
docker-compose down
```

Access:
- InfluxDB: http://localhost:8086 (admin/adminpassword123)
- Grafana: http://localhost:3000 (admin/admin)

## Configuration

### Global Settings (detector_yolo.py)

```python
TELEMETRY_ENABLED = True          # Enable InfluxDB logging
ACTIONS_ENABLED = True            # Enable keyboard shortcuts
MIN_CONFIDENCE_TO_LOG = 0.5       # Log detections above 50%
MIN_CONFIDENCE_TO_ACT = 0.75      # Trigger actions above 75%
```

These can be overridden via main.py command-line arguments.

### Gesture Actions (actions/actions.py:79-104)

Gesture mappings in `DEFAULT_ACTIONS`:
- `stop_recording_sign`: Ctrl+Shift+M (Teams/Zoom mute)
- `continue_recording_sign`: Win+H (Windows voice typing) - 2s delay
- `nail_biting_sign`: Custom notification (5s cooldown)

### Training Parameters (yolo_training/train_yolo.py:28-33)

```python
MODEL_SIZE = "yolov8n-cls"   # nano/small/medium/large/xlarge
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 16
DEVICE = 0                    # GPU index or "cpu"
```

## Key Implementation Details

### PyInstaller Bundling
- `gesture_shortcuts.spec` collects all MediaPipe/YOLO data files
- `main.py:19-23` handles `sys._MEIPASS` for bundled executables
- Set `console=False` in spec file to hide console window

### Action Debouncing
- Each gesture has configurable cooldown (`GestureAction.cooldown`)
- `_last_triggered` dict tracks last execution time per gesture
- Prevents spam from continuous detection

### Teams Integration
- `actions.py:30-53` checks for `teams.exe` / `ms-teams.exe` process
- Requires `psutil` library (gracefully degrades if unavailable)
- Only triggers microphone shortcuts when Teams is running

### Telemetry Batch Writing
- InfluxDB client uses `WriteOptions` for batching
- Default: 10 points per batch, 5s flush interval
- Singleton pattern via `get_telemetry()` function

## Dependencies

Core libraries (see `requirments.txt`):
- `mediapipe==0.10.21` - Hand detection and landmark extraction
- `ultralytics` - YOLOv8 training and inference
- `opencv-python` - Camera capture and image processing
- `pynput` - Keyboard control
- `psutil` - Process monitoring (Teams detection)
- `winotify` - Windows 10 notifications
- `influxdb-client` - Telemetry logging

## Important Notes

### Model Files
The application requires two models:
1. **MediaPipe Hand Landmarker**: `models/hand_landmarker.task`
   - Download from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

2. **YOLO Classifier**: `yolo_training/runs/gesture_classification/weights/best.pt`
   - Created by running `train_yolo.py` after collecting dataset

### Dataset Collection Workflow
1. Run `creating_dataset/signs_recording.py` for each gesture category
2. Manually change `OUTPUT_DIR_IMG` path to target gesture folder
3. Collect 100-500 samples per gesture (varied hand positions, lighting)
4. Categories must match `CATEGORIES` list in `train_yolo.py:26`

### Windows-Specific Code
- `Key.cmd` for Windows key (pynput)
- `winotify` for native Windows notifications
- `psutil` process checking for Teams.exe
- Build script (`build.bat`) uses Windows batch syntax
