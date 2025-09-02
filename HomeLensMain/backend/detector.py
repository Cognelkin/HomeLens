# backend/detector.py
from ultralytics import YOLO
import cv2

# Load YOLO model once (use "yolov8n.pt" for speed)
yolo_model = YOLO("yolov8n.pt")

def detect_furniture(image):
    """
    Run YOLOv8 on an image and return detections + crops.
    """
    results = yolo_model(image)
    detections = []

    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        label = yolo_model.names[cls_id]

        # Coordinates
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        crop = image[y1:y2, x1:x2]

        detections.append({
            "label": label,
            "crop": crop
        })

    return detections
