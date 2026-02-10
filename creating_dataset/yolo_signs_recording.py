import cv2
from ultralytics import YOLO

# 1. Load model
# 'yolov8n-pose.pt' is the "Nano" version - fastest, runs smoothly on CPU.
# On first run, the library will automatically download this file (~6 MB).
model = YOLO('yolov8n-pose.pt')

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Mirror flip (optional, for convenience)
    frame = cv2.flip(frame, 1)

    # 2. Detection and tracking (stream=True speeds up operation in loops)
    # conf=0.5 is the confidence threshold (similar to MediaPipe)
    results = model(frame, stream=True, conf=0.5)

    # YOLO returns a generator of results, we need to process it in a loop
    for result in results:
        # 3. Drawing
        # The .plot() function automatically draws skeleton on the image
        annotated_frame = result.plot()
        
        # 4. Access to coordinates (if you want to control them)
        # result.keypoints.xy returns tensor with [x, y] coordinates
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            # print(keypoints) # Uncomment to see numbers in console

        # Display
        cv2.imshow('YOLOv8 Pose', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()