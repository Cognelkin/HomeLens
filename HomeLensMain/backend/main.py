from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
import cv2
from typing import Dict, Any

from .detector import FeatureDetector
from .schemas import DetectResponse, StylePrediction, Detection
from .clarifai_client import classify_style   # NEW

import io

app = FastAPI(title="Instant Property Feature Detector", version="1.0.0")

# Allow local dev origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent  # backend/
STATIC_DIR = BASE_DIR / "static"            # backend/static/

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

detector = FeatureDetector()

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    data = await file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # Run YOLO detector
    result: Dict[str, Any] = detector.predict(img)

    # Run Clarifai style classifier
    styles = classify_style(img)

    # Convert to pydantic models
    detections = [
        Detection(
            label=d["label"],
            confidence=d["confidence"],
            box=tuple(d["box"]),
            area=d["area"]
        )
        for d in result["detections"]
    ]

    return DetectResponse(
        width=result["width"],
        height=result["height"],
        detections=detections,
        counts=result["counts"],
        amenities_score=result["amenities_score"],
        styles=styles   # NEW
    )
