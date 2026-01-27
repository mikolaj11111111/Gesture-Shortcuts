import cv2
from ultralytics import YOLO

# 1. Ładowanie modelu
# 'yolov8n-pose.pt' to wersja "Nano" - najszybsza, działa płynnie na CPU.
# Przy pierwszym uruchomieniu biblioteka sama pobierze ten plik (ok. 6 MB).
model = YOLO('yolov8n-pose.pt')

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Odbicie lustrzane (opcjonalne, dla wygody)
    frame = cv2.flip(frame, 1)

    # 2. Detekcja i śledzenie (stream=True przyspiesza działanie w pętlach)
    # conf=0.5 to próg pewności (podobnie jak w MediaPipe)
    results = model(frame, stream=True, conf=0.5)

    # YOLO zwraca generator wyników, musimy go przetworzyć w pętli
    for result in results:
        # 3. Rysowanie
        # Funkcja .plot() automatycznie rysuje szkielet na obrazie
        annotated_frame = result.plot()
        
        # 4. Dostęp do współrzędnych (jeśli chcesz nimi sterować)
        # result.keypoints.xy zwraca tensor ze współrzędnymi [x, y]
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            # print(keypoints) # Odkomentuj, aby widzieć liczby w konsoli

        # Wyświetlanie
        cv2.imshow('YOLOv8 Pose', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()