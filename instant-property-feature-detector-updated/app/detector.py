import io
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
import numpy as np
import cv2
from .condition_classifier import ConditionClassifier

# Curated set of "property features" we care about from COCO classes
PROPERTY_FEATURES = {
    "bed": 2.0,
    "couch": 1.8,          # sofa
    "chair": 0.6,
    "dining table": 1.2,
    "tv": 1.5,
    "sink": 1.3,
    "toilet": 1.6,
    "refrigerator": 1.5,
    "oven": 0.9,
    "microwave": 0.7,
    "potted plant": 0.5,
    "vase": 0.4,
    "laptop": 0.6,         # indicates a work area
}

# Model name can be changed to yolov8s, yolov8m, etc. for more accuracy
DEFAULT_MODEL = "yolov8n.pt"

class FeatureDetector:
    def __init__(self, model_name: str = DEFAULT_MODEL, conf_threshold: float = 0.25):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        # condition classifier (optional). If you have trained weights, place them at HomeLensMain/condition_classifier.pth
        try:
            self.conditioner = ConditionClassifier()
        except Exception:
            self.conditioner = None

        # Build label mapping
        self.names = self.model.model.names if hasattr(self.model.model, "names") else self.model.names

    def _filter_and_format(self, results, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        detections = []
        counts = {k: 0 for k in PROPERTY_FEATURES.keys()}

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()           # (N, 4)
            confs = r.boxes.conf.cpu().numpy()           # (N,)
            clss  = r.boxes.cls.cpu().numpy().astype(int)# (N,)

            for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
                if c < self.conf_threshold:
                    continue
                label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
                if label in PROPERTY_FEATURES:
                    area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                    detections.append({
                        "label": label,
                        "confidence": float(c),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "area": area,
                    })
                    counts[label] += 1

        return detections, counts

    def _score(self, counts: Dict[str, int], img_area: float, detections: List[Dict[str, Any]]) -> float:
        # Simple weighted sum with a size-aware bonus for large prominent items
        score = 0.0
        for label, count in counts.items():
            weight = PROPERTY_FEATURES[label]
            score += weight * count

        # Bonus: add area-based weight for large objects (e.g., big sofa/bed)
        if img_area > 0:
            large_bonus = 0.0
            for d in detections:
                rel_area = d["area"] / img_area
                if rel_area > 0.05:             # roughly >5% of image
                    large_bonus += 0.5
                if rel_area > 0.10:
                    large_bonus += 0.5
            score += large_bonus

        # Normalize roughly to 0..100
        # (heuristic cap; tweak freely)
        score = min(100.0, score * 10.0)
        return float(round(score, 2))

    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(source=image_bgr, conf=self.conf_threshold, verbose=False)
        detections, counts = self._filter_and_format(results, w, h)
        # For each detection, run the condition classifier on the cropped region (if available)
        try:
            from PIL import Image
            for d in detections:
                x1,y1,x2,y2 = map(int, d['box'])
                # clamp
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    d['condition'] = 'Unknown'
                    d['condition_score'] = 0.0
                    continue
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if getattr(self, 'conditioner', None) is not None:
                    try:
                        label, score = self.conditioner.predict(pil)
                        d['condition'] = label
                        d['condition_score'] = float(score)
                    except Exception:
                        d['condition'] = 'Unknown'
                        d['condition_score'] = 0.0
                else:
                    d['condition'] = 'Unknown'
                    d['condition_score'] = 0.0
        except Exception:
            # if PIL not available or any other error, skip condition classification
            for d in detections:
                d['condition'] = 'Unknown'
                d['condition_score'] = 0.0

        amenities_score = self._score(counts, float(w*h), detections)

        return {
            "width": w,
            "height": h,
            "detections": detections,
            "counts": counts,
            "amenities_score": amenities_score
        }
