# detector.py
import os
import time
import requests
import numpy as np
from PIL import Image
import cv2
from typing import Dict
from ultralytics import YOLO
from dotenv import load_dotenv
from base64 import b64encode

import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# Config
# -----------------------------
PROPERTY_FEATURES = {
    "bed": 2.0,
    "couch": 1.8,
    "chair": 0.6,
    "dining table": 1.2,
    "tv": 1.5,
}

DEFAULT_MODEL = "yolov8n.pt"

CLIP_LABELS = [
    "modern sofa",
    "mid-century modern sofa",
    "leather sofa",
    "fabric sofa",
    "sectional sofa",
    "minimalist sofa",
    "scandinavian sofa",
]

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

# -----------------------------
# FeatureDetector
# -----------------------------
class FeatureDetector:
    def __init__(self, model_name=DEFAULT_MODEL, conf_threshold=0.25):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.names = self.model.model.names

        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # eBay
        self.ebay_token = None
        self.token_expiry = 0
        self.ebay_endpoint = "https://api.ebay.com/buy/browse/v1/item_summary/search"

    # -----------------------------
    # PUBLIC ENTRYPOINT (IMPORTANT)
    # -----------------------------
    def detect(self, image: Image.Image) -> Dict:
        """
        Unified detect method used by /detect and /shop
        Accepts PIL Image, converts internally
        """
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.predict(image_bgr)

    # -----------------------------
    # eBay OAuth
    # -----------------------------
    def _get_ebay_token(self):
        if self.ebay_token and time.time() < self.token_expiry - 60:
            return self.ebay_token

        auth = b64encode(f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        }

        r = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers=headers,
            data=data,
        )
        r.raise_for_status()

        token_data = r.json()
        self.ebay_token = token_data["access_token"]
        self.token_expiry = time.time() + token_data["expires_in"]
        return self.ebay_token

    # -----------------------------
    # YOLO helpers
    # -----------------------------
    def _filter_and_format(self, results, img_w, img_h):
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
            ):
                if conf < self.conf_threshold:
                    continue

                label = self.names[cls_id]
                if label not in PROPERTY_FEATURES:
                    continue

                detections.append({
                    "id": len(detections),
                    "label": label,
                    "confidence": float(conf),
                    "box": [float(v) for v in box],
                })

        return detections

    def crop_object(self, image_bgr, box):
        x1, y1, x2, y2 = map(int, box)
        return image_bgr[y1:y2, x1:x2]

    # -----------------------------
    # SHOP FOR SELECTED DETECTION
    # -----------------------------
    def shop_for_detection(self, image: Image.Image, detection_id: int):
        result = self.detect(image)
        detections = result["detections"]

        if detection_id < 0 or detection_id >= len(detections):
            raise ValueError("Invalid detection_id")

        det = detections[detection_id]
        label = det["label"]

        # Simple query (you’ll enrich this later)
        query = f"modern {label} furniture"

        items = self.search_ebay(query)

        return {
            "detection_id": detection_id,
            "label": label,
            "query": query,
            "items": items,
        }

    # -----------------------------
    # CLIP style detection
    # -----------------------------
    def classify_style(self, crop_bgr):
        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.clip_processor(
            text=CLIP_LABELS,
            images=image_rgb,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        best_idx = probs.argmax().item()
        return CLIP_LABELS[best_idx]

    # -----------------------------
    # eBay search (Browse API)
    # -----------------------------
    def search_ebay(self, query, limit=6):
        token = self._get_ebay_token()
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": query, "limit": limit}

        r = requests.get(self.ebay_endpoint, headers=headers, params=params)
        r.raise_for_status()
        return r.json().get("itemSummaries", [])

    # -----------------------------
    # CORE PREDICTION
    # -----------------------------
    def predict(self, image_bgr: np.ndarray, selected_label: str = None):
        h, w = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)

        detections = self._filter_and_format(results, w, h)

        recommendations = []
        if selected_label:
            matches = [d for d in detections if d["label"] == selected_label]
            if matches:
                crop = self.crop_object(image_bgr, matches[0]["box"])
                style = self.classify_style(crop)
                recommendations = self.search_ebay(style)

        return {
            "width": w,
            "height": h,
            "detections": detections,
            "recommendations": recommendations,
        }
