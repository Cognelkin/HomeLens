# import io
# from typing import List, Dict, Any, Tuple
# from ultralytics import YOLO
# import numpy as np
# import cv2
#
# # Curated set of "property features" we care about from COCO classes
# PROPERTY_FEATURES = {
#     "bed": 2.0,
#     "couch": 1.8,          # sofa
#     "chair": 0.6,
#     "dining table": 1.2,
#     "tv": 1.5,
#     "sink": 1.3,
#     "toilet": 1.6,
#     "refrigerator": 1.5,
#     "oven": 0.9,
#     "microwave": 0.7,
#     "potted plant": 0.5,
#     "vase": 0.4,
#     "laptop": 0.6,         # indicates a work area
# }
#
# # Model name can be changed to yolov8s, yolov8m, etc. for more accuracy
# DEFAULT_MODEL = "yolov8n.pt"
#
# class FeatureDetector:
#     def __init__(self, model_name: str = DEFAULT_MODEL, conf_threshold: float = 0.25):
#         self.model = YOLO(model_name)
#         self.conf_threshold = conf_threshold
#         # Build label mapping
#         self.names = self.model.model.names if hasattr(self.model.model, "names") else self.model.names
#
#     def _filter_and_format(self, results, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
#         detections = []
#         counts = {k: 0 for k in PROPERTY_FEATURES.keys()}
#
#         for r in results:
#             if r.boxes is None:
#                 continue
#             boxes = r.boxes.xyxy.cpu().numpy()           # (N, 4)
#             confs = r.boxes.conf.cpu().numpy()           # (N,)
#             clss  = r.boxes.cls.cpu().numpy().astype(int)# (N,)
#
#             for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
#                 if c < self.conf_threshold:
#                     continue
#                 label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
#                 if label in PROPERTY_FEATURES:
#                     area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
#                     detections.append({
#                         "label": label,
#                         "confidence": float(c),
#                         "box": [float(x1), float(y1), float(x2), float(y2)],
#                         "area": area,
#                     })
#                     counts[label] += 1
#
#         return detections, counts
#
#     def _score(self, counts: Dict[str, int], img_area: float, detections: List[Dict[str, Any]]) -> float:
#         # Simple weighted sum with a size-aware bonus for large prominent items
#         score = 0.0
#         for label, count in counts.items():
#             weight = PROPERTY_FEATURES[label]
#             score += weight * count
#
#         # Bonus: add area-based weight for large objects (e.g., big sofa/bed)
#         if img_area > 0:
#             large_bonus = 0.0
#             for d in detections:
#                 rel_area = d["area"] / img_area
#                 if rel_area > 0.05:             # roughly >5% of image
#                     large_bonus += 0.5
#                 if rel_area > 0.10:
#                     large_bonus += 0.5
#             score += large_bonus
#
#         # Normalize roughly to 0..100
#         score = min(100.0, score * 10.0)
#         return float(round(score, 2))
#
#     def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
#         h, w = image_bgr.shape[:2]
#         results = self.model.predict(source=image_bgr, conf=self.conf_threshold, verbose=False)
#         detections, counts = self._filter_and_format(results, w, h)
#         amenities_score = self._score(counts, float(w*h), detections)
#
#         return {
#             "width": w,
#             "height": h,
#             "detections": detections,
#             "counts": counts,
#             "amenities_score": amenities_score
#         }
#
#     def crop_objects(self, image_bgr: np.ndarray, detections: list):
#         """Crop objects based on YOLO detections and return dict of crops."""
#         crops = []
#         for det in detections:
#             x1, y1, x2, y2 = map(int, det["box"])
#             crop = image_bgr[y1:y2, x1:x2]
#             if crop.size > 0:
#                 crops.append({
#                     "label": det["label"],
#                     "confidence": det["confidence"],
#                     "crop": crop
#                 })
#         return crops

# import os
# import requests
# import numpy as np
# from ultralytics import YOLO
# from dotenv import load_dotenv
# import time
# import requests
# import cv2
# from typing import List, Dict, Any, Tuple
#
# PROPERTY_FEATURES = {
#     "bed": 2.0,
#     "couch": 1.8,
#     "chair": 0.6,
#     "dining table": 1.2,
#     "tv": 1.5,
#     "sink": 1.3,
#     "toilet": 1.6,
#     "refrigerator": 1.5,
#     "oven": 0.9,
#     "microwave": 0.7,
#     "potted plant": 0.5,
#     "vase": 0.4,
#     "laptop": 0.6,
# }
#
# DEFAULT_MODEL = "yolov8n.pt"
# dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
# load_dotenv(dotenv_path)
#
# CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
# CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
# TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"

# print("Testing here")
# url = "https://svcs.ebay.com/services/search/FindingService/v1"
# params = {
#     "OPERATION-NAME": "findItemsByKeywords",
#     "SERVICE-VERSION": "1.0.0",
#     "SECURITY-APPNAME": EBAY_APP_ID,
#     "RESPONSE-DATA-FORMAT": "JSON",
#     "REST-PAYLOAD": "",
#     "keywords": "sofa",
# }
# r = requests.get(url, params=params)
# print(r.json())

# detector.py
import io
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
import numpy as np
import cv2
from base64 import b64encode
import time

# -----------------------------
# Property features and YOLO
# -----------------------------
PROPERTY_FEATURES = {
    "bed": 2.0,
    "couch": 1.8,
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
    "laptop": 0.6,
}

DEFAULT_MODEL = "yolov8n.pt"

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"


class FeatureDetector:
    def __init__(self, model_name: str = DEFAULT_MODEL, conf_threshold: float = 0.25):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.names = self.model.model.names if hasattr(self.model.model, "names") else self.model.names

        # eBay API
        self.ebay_client_id = EBAY_CLIENT_ID  # set in .env
        self.ebay_client_secret = EBAY_CLIENT_SECRET
        self.ebay_token = None
        self.token_expiry = 0
        self.ebay_endpoint = "https://api.ebay.com/buy/browse/v1/item_summary/search"

    def _get_ebay_token(self) -> str:
        if self.ebay_token and time.time() < self.token_expiry - 60:
            return self.ebay_token  # reuse existing token if still valid

        auth = b64encode(f"{self.ebay_client_id}:{self.ebay_client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
        res = requests.post("https://api.ebay.com/identity/v1/oauth2/token", headers=headers, data=data)
        res.raise_for_status()
        token_info = res.json()
        self.ebay_token = token_info["access_token"]
        self.token_expiry = time.time() + int(token_info.get("expires_in", 7200))
        return self.ebay_token

    # -----------------------------
    # YOLO detection functions
    # -----------------------------
    def _filter_and_format(self, results, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        detections = []
        counts = {k: 0 for k in PROPERTY_FEATURES.keys()}

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

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
        score = 0.0
        for label, count in counts.items():
            weight = PROPERTY_FEATURES[label]
            score += weight * count

        if img_area > 0:
            large_bonus = 0.0
            for d in detections:
                rel_area = d["area"] / img_area
                if rel_area > 0.05:
                    large_bonus += 0.5
                if rel_area > 0.10:
                    large_bonus += 0.5
            score += large_bonus

        score = min(100.0, score * 10.0)
        return float(round(score, 2))

    # -----------------------------
    # eBay Browse API search
    # -----------------------------
    def search_ebay(self, query: str, entries: int = 5) -> List[Dict[str, Any]]:
        try:
            token = self._get_ebay_token()
        except Exception as e:
            print(f"[eBay] Error fetching OAuth token: {e}")
            return []

        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": query, "limit": entries}

        try:
            res = requests.get(self.ebay_endpoint, headers=headers, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            items = data.get("itemSummaries", [])
            return items
        except Exception as e:
            print(f"[eBay] Error fetching '{query}': {e}")
            return []
    # -----------------------------
    # Main predict function
    # -----------------------------
    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(source=image_bgr, conf=self.conf_threshold, verbose=False)
        detections, counts = self._filter_and_format(results, w, h)
        amenities_score = self._score(counts, float(w * h), detections)

        # Hardcode "sofa" for testing if no detections
        keywords = [d["label"] for d in detections] or ["sofa"]
        recommendations = []
        for k in keywords:
            recommendations.extend(self.search_ebay(k))

        return {
            "width": w,
            "height": h,
            "detections": detections,
            "counts": counts,
            "amenities_score": amenities_score,
            "recommendations": recommendations
        }