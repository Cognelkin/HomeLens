# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, Any
# import numpy as np
# import cv2
# from .detector import FeatureDetector
# from .schemas import DetectResponse, Detection
#
# app = FastAPI(title="HomeLens Property Detector", version="1.0.0")
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Serve static frontend
# app.mount("/static", StaticFiles(directory="backend/static"), name="static")
#
# @app.get("/", response_class=HTMLResponse)
# async def index():
#     with open("backend/static/index.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(f.read())
#
# detector = FeatureDetector()
#
# @app.post("/detect", response_model=DetectResponse)
# async def detect(file: UploadFile = File(...)) -> DetectResponse:
#     data = await file.read()
#     file_bytes = np.frombuffer(data, np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Invalid image")
#     result: Dict[str, Any] = detector.predict(img)
#
#     detections = [
#         Detection(label=d["label"], confidence=d["confidence"], box=tuple(d["box"]), area=d["area"])
#         for d in result["detections"]
#     ]
#
#     # Use detected labels for eBay search; for now, just first label or "sofa" as fallback
#     query = detections[0].label if detections else "sofa"
#     recommendations = detector.search_ebay(query)
#
#     return DetectResponse(
#         width=result["width"],
#         height=result["height"],
#         detections=detections,
#         counts=result["counts"],
#         amenities_score=result["amenities_score"],
#         recommendations=recommendations
#     )

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from .detector import FeatureDetector

app = FastAPI(title="HomeLens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

detector = FeatureDetector()


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("backend/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    selected_label: str | None = Form(None),
):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    result = detector.predict(image, selected_label)
    return result
