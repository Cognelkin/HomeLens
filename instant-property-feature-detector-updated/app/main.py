from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from typing import Dict, Any
from .detector import FeatureDetector
from .schemas import DetectResponse, Detection
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

app.mount("/frontend", StaticFiles(directory="HomeLensMain/frontend"), name="frontend")
detector = FeatureDetector()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("HomeLensMain/frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    data = await file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    result: Dict[str, Any] = detector.predict(img)

    # Convert to pydantic models
    detections = [
        Detection(label=d["label"], confidence=d["confidence"], box=tuple(d["box"]), area=d["area"], condition=d.get("condition","Unknown"), condition_score=d.get("condition_score",0.0))
        for d in result["detections"]
    ]
    return DetectResponse(
        width=result["width"],
        height=result["height"],
        detections=detections,
        counts=result["counts"],
        amenities_score=result["amenities_score"]
    )
