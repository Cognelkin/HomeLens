from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained COCO model (can be swapped for custom-trained furniture model later)
model = YOLO("yolov8n.pt")

def detect_objects(img):
    results = model(img)
    detections = []
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        conf = float(r.conf[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        detections.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
    return detections
