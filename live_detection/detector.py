import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import torch
import os
import sys
import time
from collections import defaultdict

# Add path to project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gesture_mlp import load_model, normalize_keypoints
from telemetry import get_telemetry
from actions import GestureActionHandler

# --- PATHS ---
HAND_LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

# --- CONFIGURATION ---
TELEMETRY_ENABLED = True  # Disable if you don't have InfluxDB running
ACTIONS_ENABLED = False  # Disable if you don't want to execute keyboard shortcuts
MIN_CONFIDENCE_TO_LOG = 0.5  # Minimum confidence to log detection
MIN_CONFIDENCE_TO_ACT = 0.75  # Minimum confidence to execute action


def main():
    # Load MLP model
    print("Loading model...")
    model, categories = load_model()
    print(f"Categories: {categories}")

    # Initialize telemetry
    telemetry = get_telemetry(enabled=TELEMETRY_ENABLED)
    if telemetry.is_connected:
        print("Telemetry: CONNECTED to InfluxDB")
    else:
        print("Telemetry: OFFLINE (data will not be saved)")

    # Initialize actions (keyboard shortcuts)
    action_handler = GestureActionHandler(
        min_confidence=MIN_CONFIDENCE_TO_ACT,
        enabled=ACTIONS_ENABLED
    )
    if ACTIONS_ENABLED:
        print("Actions: ENABLED")
        print("  - stop_recording_sign → Ctrl+Shift+M (mute microphone)")
        print("  - continue_recording_sign → Ctrl+Shift+M (unmute microphone)")
        print("  - nail_biting_sign → Notification")
    else:
        print("Actions: DISABLED")

    # MediaPipe Hand Landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    camera = cv2.VideoCapture(0)
    start_time = time.time()

    # Session statistics
    gesture_counts = defaultdict(int)
    frame_count = 0

    # Activity time tracking (e.g., nail_biting)
    current_activity = None
    activity_start_time = None

    # Last executed action (for display)
    last_action_text = ""
    last_action_time = 0

    print("Live detection started. Q - exit")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Hand detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            gesture_text = "No hand"
            confidence = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                # Get keypoints and normalize
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                keypoints_normalized = normalize_keypoints(keypoints)

                # Prediction
                with torch.no_grad():
                    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32).unsqueeze(0)
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    gesture_text = categories[predicted.item()]
                    confidence = confidence.item()

                # --- TELEMETRY: Log detection ---
                if confidence >= MIN_CONFIDENCE_TO_LOG:
                    telemetry.log_detection(gesture_text, confidence)
                    gesture_counts[gesture_text] += 1

                    # --- ACTIONS: Execute keyboard shortcut ---
                    if action_handler.trigger(gesture_text, confidence):
                        action_info = action_handler.get_action_info(gesture_text)
                        last_action_text = action_info or gesture_text
                        last_action_time = time.time()

                    # Activity time tracking
                    if gesture_text != current_activity:
                        # End previous activity
                        if current_activity and activity_start_time:
                            duration = time.time() - activity_start_time
                            if duration > 1.0:  # Ignore shorter than 1s
                                telemetry.log_activity_duration(current_activity, duration)

                        # Start new activity
                        current_activity = gesture_text
                        activity_start_time = time.time()

                # Draw landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )
            else:
                # No hand - end activity
                if current_activity and activity_start_time:
                    duration = time.time() - activity_start_time
                    if duration > 1.0:
                        telemetry.log_activity_duration(current_activity, duration)
                    current_activity = None
                    activity_start_time = None

            # Display prediction
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"{gesture_text} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Telemetry status
            status = "DB: ON" if telemetry.is_connected else "DB: OFF"
            cv2.putText(frame, status, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Last executed action (show for 3 seconds)
            if last_action_text and (time.time() - last_action_time) < 3.0:
                cv2.putText(frame, f"Action: {last_action_text}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow('Gesture Detection - Q: exit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- TELEMETRY: Final session statistics ---
    session_duration = time.time() - start_time
    avg_fps = frame_count / session_duration if session_duration > 0 else 0
    telemetry.log_session_stats(dict(gesture_counts), session_duration, avg_fps)
    telemetry.flush()

    print(f"\n--- Session statistics ---")
    print(f"Time: {session_duration:.1f}s | FPS: {avg_fps:.1f}")
    print(f"Detections: {dict(gesture_counts)}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()