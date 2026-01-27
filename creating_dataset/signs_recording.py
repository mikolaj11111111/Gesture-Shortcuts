import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Importujemy dodatkowe moduły do rysowania i formatowania danych
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
OUTPUT_DIR_IMG = os.path.join(os.path.dirname(__file__), 'data', 'other', 'images')
KEYPOINTS_CSV = os.path.join(os.path.dirname(__file__), 'data', 'other', 'keypoints.csv')
os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
os.makedirs(os.path.dirname(KEYPOINTS_CSV), exist_ok=True)

MARGIN = 20
BBOX_WIDTH, BBOX_HEIGHT = 256, 256

frame_counter = 0
recording = False  # Tryb automatycznego nagrywania
camera = cv2.VideoCapture(0)


def get_hand_bounding_box(hand_landmarks, frame_width, frame_height, margin=MARGIN):
    """Oblicza bounding box dłoni z landmarków z marginesem."""
    x_coords = [landmark.x for landmark in hand_landmarks]
    y_coords = [landmark.y for landmark in hand_landmarks]

    # Konwersja ze współrzędnych znormalizowanych (0-1) na piksele
    x_min = int(min(x_coords) * frame_width) - margin
    x_max = int(max(x_coords) * frame_width) + margin
    y_min = int(min(y_coords) * frame_height) - margin
    y_max = int(max(y_coords) * frame_height) + margin

    # Ograniczenie do wymiarów obrazu
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)

    return x_min, y_min, x_max, y_max

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3, 
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3)

# Tworzymy instancję HandLandmarker
with HandLandmarker.create_from_options(options) as landmarker:
    start_time = time.time()
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Odbicie lustrzane dla naturalnego odczucia
        frame = cv2.flip(frame, 1)
        
        # Konwersja do formatu MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detekcja
        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        hand_landmarker_result = landmarker.detect_for_video(mp_image,frame_timestamp_ms)

        # --- WYCINANIE I RYSOWANIE ---
        frame_height, frame_width = frame.shape[:2]
        

        # Listy do zapisu (obrazy i keypoints)
        hand_data = []  # [(crop, keypoints), ...]

        if hand_landmarker_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_landmarker_result.hand_landmarks):

                # 1. Oblicz bounding box i wytnij dłoń PRZED rysowaniem
                x_min, y_min, x_max, y_max = get_hand_bounding_box(
                    hand_landmarks, frame_width, frame_height
                )
                hand_crop = frame[y_min:y_max, x_min:x_max].copy()  # copy() - czysta kopia bez keypointów
                print(frame.shape, hand_crop.shape)
                # 2. Zapisz keypoints jako numpy array (21 punktów × 3 współrzędne)
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

                if hand_crop.size > 0:
                    hand_data.append((hand_crop, keypoints))

                # 3. Rysuj bounding box na podglądzie
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # 4. Konwersja wyników nowego API na format zrozumiały dla drawing_utils
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # 5. Rysowanie punktów i połączeń na klatce (frame) - tylko do podglądu
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

                # 6. Pokaż podgląd wycinka
                if hand_crop.size > 0:
                    cv2.imshow(f'Hand {hand_idx}', hand_crop)

        # Indykator nagrywania
        if recording:
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)  # Czerwone kółko
            cv2.putText(frame, f"REC ({frame_counter})", (55, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('R: nagrywaj, S: zapisz, Q: wyjdz', frame)

        # Obsługa klawiszy
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print(f"Nagrywanie: {'ON' if recording else 'OFF'}")
        elif key == ord('s') and hand_data:
            # Ręczny zapis pojedynczej klatki
            for crop, keypoints in hand_data:
                filename = f"{frame_counter:05d}"
                crop_resized = cv2.resize(crop, (BBOX_WIDTH, BBOX_HEIGHT))
                img_path = os.path.join(OUTPUT_DIR_IMG, f"{filename}.jpg")
                cv2.imwrite(img_path, crop_resized)
                with open(KEYPOINTS_CSV, 'a') as f:
                    row = keypoints.flatten()
                    f.write(','.join(map(str, row)) + '\n')
                print(f"Zapisano: {img_path} | CSV row #{frame_counter}")
                frame_counter += 1

        # Automatyczny zapis gdy nagrywanie włączone
        if recording and hand_data:
            for crop, keypoints in hand_data:
                filename = f"{frame_counter:05d}"
                crop_resized = cv2.resize(crop, (BBOX_WIDTH, BBOX_HEIGHT))
                img_path = os.path.join(OUTPUT_DIR_IMG, f"{filename}.jpg")
                cv2.imwrite(img_path, crop_resized)
                with open(KEYPOINTS_CSV, 'a') as f:
                    row = keypoints.flatten()
                    f.write(','.join(map(str, row)) + '\n')
                frame_counter += 1

    camera.release()
    cv2.destroyAllWindows()