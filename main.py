"""
Gesture Shortcuts - Main Entry Point

This is the main entry point for the Gesture Shortcuts application.
Run this file to start gesture detection with YOLO classification.

Usage:
    python main.py              # Normal mode (with console)
    python main.py --no-actions # Disable keyboard actions
    python main.py --no-telemetry # Disable telemetry logging
"""

import sys
import os
import io
import argparse

# Fix console encoding for PyInstaller bundles on Windows (cp1250 can't handle Unicode)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the project root is in the path
# When running as PyInstaller bundle, use _MEIPASS as base path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    PROJECT_ROOT = sys._MEIPASS
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def is_frozen():
    """Check if running as a PyInstaller bundled executable."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def check_dependencies():
    """Check if all required dependencies are installed."""
    # Skip dependency checks when running as bundled .exe
    # All dependencies are already bundled by PyInstaller
    if is_frozen():
        return True
    
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")
    
    try:
        from pynput.keyboard import Controller
    except ImportError:
        missing.append("pynput")
    
    if missing:
        print("=" * 50)
        print("ERROR: Missing required dependencies!")
        print("=" * 50)
        print("\nPlease install the following packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nOr install all at once:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 50)
        return False
    
    return True


def check_models():
    """Check if required model files exist."""
    models_ok = True
    
    # Hand landmarker model
    hand_model = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    if not os.path.exists(hand_model):
        print(f"WARNING: Hand landmarker model not found: {hand_model}")
        print("  Download from: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
        models_ok = False
    
    # YOLO classification model
    yolo_model = os.path.join(PROJECT_ROOT, 'yolo_training', 'runs', 'gesture_classification', 'weights', 'best.pt')
    if not os.path.exists(yolo_model):
        print(f"WARNING: YOLO model not found: {yolo_model}")
        print("  Run first: python yolo_training/train_yolo.py")
        models_ok = False
    
    return models_ok


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gesture Shortcuts - Control your computer with hand gestures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start with all features enabled
  python main.py --no-actions       Disable keyboard shortcuts
  python main.py --no-telemetry     Disable InfluxDB logging
  python main.py --confidence 0.8   Set minimum confidence to 80%
        """
    )
    
    parser.add_argument(
        '--no-actions', 
        action='store_true',
        help='Disable keyboard action triggers'
    )
    
    parser.add_argument(
        '--no-telemetry', 
        action='store_true',
        help='Disable telemetry logging to InfluxDB'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.75,
        help='Minimum confidence threshold for actions (default: 0.75)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    
    return parser.parse_args()


def print_banner():
    """Print application banner."""
    print()
    print("=" * 50)
    print("  🖐️  GESTURE SHORTCUTS")
    print("  Control your computer with hand gestures")
    print("=" * 50)
    print()


def main():
    """Main application entry point."""
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Check dependencies
    print("[1/3] Checking dependencies...")
    if not check_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)
    print("      ✓ All dependencies OK")
    
    # Check models
    print("[2/3] Checking models...")
    if not check_models():
        print("\n      ⚠ Some models are missing. Detection may not work correctly.")
    else:
        print("      ✓ All models found")
    
    # Start detection
    print("[3/3] Starting gesture detection...")
    print()
    print(f"  Actions:    {'ENABLED' if not args.no_actions else 'DISABLED'}")
    print(f"  Telemetry:  {'ENABLED' if not args.no_telemetry else 'DISABLED'}")
    print(f"  Confidence: {args.confidence:.0%}")
    print(f"  Camera:     {args.camera}")
    print()
    print("  Press 'Q' in the camera window to exit")
    print("-" * 50)
    print()
    
    # Import and run detector
    try:
        from live_detection import detector_yolo
        
        # Override configuration based on arguments
        detector_yolo.TELEMETRY_ENABLED = not args.no_telemetry
        detector_yolo.ACTIONS_ENABLED = not args.no_actions
        detector_yolo.MIN_CONFIDENCE_TO_ACT = args.confidence
        
        # Run the main detection loop
        detector_yolo.main()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\nGesture Shortcuts closed. Goodbye! 👋")


if __name__ == '__main__':
    main()
