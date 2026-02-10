# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Gesture Shortcuts application.

Build command:
    pyinstaller gesture_shortcuts.spec

Output:
    dist/GestureShortcuts/ - folder with executable and dependencies
"""

import os
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs, collect_all

# Project root directory
PROJECT_ROOT = os.path.abspath('.')

# Collect MediaPipe - use collect_all to get everything (data, binaries, submodules)
mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all('mediapipe')

# Explicitly collect ALL .pyd and .dll files from mediapipe (PyInstaller misses these)
import mediapipe as _mp
_mp_dir = os.path.dirname(_mp.__file__)
_mp_native_files = []
for ext in ('*.pyd', '*.dll'):
    for filepath in glob.glob(os.path.join(_mp_dir, '**', ext), recursive=True):
        rel_dir = os.path.relpath(os.path.dirname(filepath), os.path.dirname(_mp_dir))
        _mp_native_files.append((filepath, rel_dir))

# Collect OpenCV data files (including config.py)
cv2_datas = collect_data_files('cv2')

# Collect ultralytics data files
ultralytics_datas = collect_data_files('ultralytics')
ultralytics_hiddenimports = collect_submodules('ultralytics')

a = Analysis(
    ['main.py'],
    pathex=[PROJECT_ROOT],
    binaries=_mp_native_files + mediapipe_binaries,
    datas=[
        # Include model files
        ('models/hand_landmarker.task', 'models'),
        ('yolo_training/runs/gesture_classification/weights/best.pt', 'yolo_training/runs/gesture_classification/weights'),
        # Include any other required data
    ] + mediapipe_datas + ultralytics_datas + cv2_datas,
    hiddenimports=[
        # MediaPipe dependencies (auto-collected)
    ] + mediapipe_hiddenimports + ultralytics_hiddenimports + [
        # Explicit MediaPipe modules
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python._framework_bindings',
        'mediapipe.python.solutions',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
        'mediapipe.framework',
        'mediapipe.framework.formats',
        'mediapipe.framework.formats.landmark_pb2',
        # OpenCV
        'cv2',
        # Ultralytics/YOLO
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.utils',
        # Project modules
        'actions',
        'actions.actions',
        'telemetry',
        'telemetry.telemetry',
        'live_detection',
        'live_detection.detector_yolo',
        # Other dependencies
        'pynput',
        'pynput.keyboard',
        'psutil',
        'winotify',
        'numpy',
        'PIL',
        # Matplotlib - required by mediapipe.python.solutions.drawing_utils
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'pytest',
        'jupyter',
        'notebook',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GestureShortcuts',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Uncomment when you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GestureShortcuts',
)
