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

# Dodaj ścieżkę do modułów projektu
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gesture_mlp import load_model, normalize_keypoints
from telemetry import get_telemetry
from actions import GestureActionHandler

# --- ŚCIEŻKI ---
HAND_LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

# --- KONFIGURACJA ---
TELEMETRY_ENABLED = True  # Wyłącz jeśli nie masz uruchomionego InfluxDB
ACTIONS_ENABLED = False  # Wyłącz jeśli nie chcesz wykonywać skrótów klawiszowych
MIN_CONFIDENCE_TO_LOG = 0.5  # Minimalna pewność aby logować detekcję
MIN_CONFIDENCE_TO_ACT = 0.75  # Minimalna pewność aby wykonać akcję


def main():
    # Wczytaj model MLP
    print("Wczytywanie modelu...")
    model, categories = load_model()
    print(f"Kategorie: {categories}")

    # Inicjalizacja telemetrii
    telemetry = get_telemetry(enabled=TELEMETRY_ENABLED)
    if telemetry.is_connected:
        print("Telemetria: POŁĄCZONO z InfluxDB")
    else:
        print("Telemetria: OFFLINE (dane nie będą zapisywane)")

    # Inicjalizacja akcji (skróty klawiszowe)
    action_handler = GestureActionHandler(
        min_confidence=MIN_CONFIDENCE_TO_ACT,
        enabled=ACTIONS_ENABLED
    )
    if ACTIONS_ENABLED:
        print("Akcje: WŁĄCZONE")
        print("  - stop_recording_sign → Ctrl+Shift+M (wycisz mikrofon)")
        print("  - continue_recording_sign → Ctrl+Shift+M (włącz mikrofon)")
        print("  - nail_biting_sign → Powiadomienie")
    else:
        print("Akcje: WYŁĄCZONE")

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

    # Statystyki sesji
    gesture_counts = defaultdict(int)
    frame_count = 0

    # Śledzenie czasu aktywności (np. nail_biting)
    current_activity = None
    activity_start_time = None

    # Ostatnia wykonana akcja (do wyświetlenia)
    last_action_text = ""
    last_action_time = 0

    print("Uruchomiono live detection. Q - wyjscie")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Detekcja dłoni
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            gesture_text = "Brak dloni"
            confidence = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                # Pobierz keypoints i normalizuj
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                keypoints_normalized = normalize_keypoints(keypoints)

                # Predykcja
                with torch.no_grad():
                    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32).unsqueeze(0)
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    gesture_text = categories[predicted.item()]
                    confidence = confidence.item()

                # --- TELEMETRIA: Logowanie detekcji ---
                if confidence >= MIN_CONFIDENCE_TO_LOG:
                    telemetry.log_detection(gesture_text, confidence)
                    gesture_counts[gesture_text] += 1

                    # --- AKCJE: Wykonaj skrót klawiszowy ---
                    if action_handler.trigger(gesture_text, confidence):
                        action_info = action_handler.get_action_info(gesture_text)
                        last_action_text = action_info or gesture_text
                        last_action_time = time.time()

                    # Śledzenie czasu aktywności
                    if gesture_text != current_activity:
                        # Zakończ poprzednią aktywność
                        if current_activity and activity_start_time:
                            duration = time.time() - activity_start_time
                            if duration > 1.0:  # Ignoruj krótsze niż 1s
                                telemetry.log_activity_duration(current_activity, duration)

                        # Rozpocznij nową aktywność
                        current_activity = gesture_text
                        activity_start_time = time.time()

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
                # Brak dłoni - zakończ aktywność
                if current_activity and activity_start_time:
                    duration = time.time() - activity_start_time
                    if duration > 1.0:
                        telemetry.log_activity_duration(current_activity, duration)
                    current_activity = None
                    activity_start_time = None

            # Wyświetl predykcję
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"{gesture_text} ({confidence:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Status telemetrii
            status = "DB: ON" if telemetry.is_connected else "DB: OFF"
            cv2.putText(frame, status, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Ostatnia wykonana akcja (pokazuj przez 3 sekundy)
            if last_action_text and (time.time() - last_action_time) < 3.0:
                cv2.putText(frame, f"Akcja: {last_action_text}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow('Gesture Detection - Q: wyjdz', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- TELEMETRIA: Statystyki końcowe sesji ---
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