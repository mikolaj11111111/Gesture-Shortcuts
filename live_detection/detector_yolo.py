import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import sys
import time
from collections import defaultdict
from ultralytics import YOLO

# Dodaj ścieżkę do modułów projektu
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from telemetry import get_telemetry
from actions import GestureActionHandler

# --- ŚCIEŻKI ---
HAND_LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'yolo_training', 'runs', 'gesture_classification', 'weights', 'best.pt')

# --- KONFIGURACJA ---
TELEMETRY_ENABLED = True
ACTIONS_ENABLED = True
MIN_CONFIDENCE_TO_LOG = 0.5
MIN_CONFIDENCE_TO_ACT = 0.75


def main():
    # Wczytaj model YOLO Classification
    print("Wczytywanie modelu YOLO...")
    print(f"Ścieżka: {YOLO_MODEL_PATH}")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono modelu YOLO!")
        print(f"Uruchom najpierw: python yolo_training/train_yolo.py")
        return

    print("Ładowanie modelu...")    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    categories = yolo_model.names  # {0: 'continue_recording_sign', 1: 'nail_biting_sign', ...}
    print(f"Kategorie: {list(categories.values())}")

    # Inicjalizacja telemetrii
    telemetry = get_telemetry(enabled=TELEMETRY_ENABLED)
    if telemetry.is_connected:
        print("Telemetria: POŁĄCZONO z InfluxDB")
    else:
        print("Telemetria: OFFLINE")

    # Inicjalizacja akcji
    action_handler = GestureActionHandler(
        min_confidence=MIN_CONFIDENCE_TO_ACT,
        enabled=ACTIONS_ENABLED
    )
    if ACTIONS_ENABLED:
        print("Akcje: WŁĄCZONE")
    else:
        print("Akcje: WYŁĄCZONE")

    # MediaPipe Hand Landmarker (do wykrywania i wycinania dłoni)
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

    # Statystyki sesji
    gesture_counts = defaultdict(int)
    frame_count = 0
    current_activity = None
    activity_start_time = None
    last_action_text = ""
    last_action_time = 0

    print("Uruchomiono YOLO detection. Q - wyjscie")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]

            # Detekcja dłoni przez MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            gesture_text = "Brak dloni"
            confidence = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                # Oblicz bounding box dłoni
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                
                margin = 20
                x_min = max(0, int(min(x_coords) * frame_width) - margin)
                x_max = min(frame_width, int(max(x_coords) * frame_width) + margin)
                y_min = max(0, int(min(y_coords) * frame_height) - margin)
                y_max = min(frame_height, int(max(y_coords) * frame_height) + margin)

                # Wytnij dłoń
                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size > 0:
                    # Klasyfikacja przez YOLO
                    results = yolo_model.predict(hand_crop, verbose=False)
                    
                    if results and len(results) > 0:
                        probs = results[0].probs
                        if probs is not None:
                            top1_idx = probs.top1
                            confidence = probs.top1conf.item()
                            gesture_text = categories[top1_idx]

                # --- TELEMETRIA ---
                if confidence >= MIN_CONFIDENCE_TO_LOG:
                    telemetry.log_detection(gesture_text, confidence)
                    gesture_counts[gesture_text] += 1

                    # --- AKCJE ---
                    if action_handler.trigger(gesture_text, confidence):
                        action_info = action_handler.get_action_info(gesture_text)
                        last_action_text = action_info or gesture_text
                        last_action_time = time.time()

                    # Śledzenie aktywności
                    if gesture_text != current_activity:
                        if current_activity and activity_start_time:
                            duration = time.time() - activity_start_time
                            if duration > 1.0:
                                telemetry.log_activity_duration(current_activity, duration)
                        current_activity = gesture_text
                        activity_start_time = time.time()

                # Rysuj bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Rysuj landmarks
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
                # Brak dłoni
                if current_activity and activity_start_time:
                    duration = time.time() - activity_start_time
                    if duration > 1.0:
                        telemetry.log_activity_duration(current_activity, duration)
                    current_activity = None
                    activity_start_time = None

            # Wyświetl predykcję
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"YOLO: {gesture_text} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Status
            status = "DB: ON" if telemetry.is_connected else "DB: OFF"
            cv2.putText(frame, status, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Ostatnia akcja
            if last_action_text and (time.time() - last_action_time) < 3.0:
                cv2.putText(frame, f"Akcja: {last_action_text}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow('YOLO Gesture Detection - Q: wyjdz', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Statystyki końcowe
    session_duration = time.time() - start_time
    avg_fps = frame_count / session_duration if session_duration > 0 else 0
    telemetry.log_session_stats(dict(gesture_counts), session_duration, avg_fps)
    telemetry.flush()

    print(f"\n--- Statystyki sesji ---")
    print(f"Czas: {session_duration:.1f}s | FPS: {avg_fps:.1f}")
    print(f"Detekcje: {dict(gesture_counts)}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
