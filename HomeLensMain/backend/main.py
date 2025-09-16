# # from pathlib import Path
# #
# # from fastapi import FastAPI, UploadFile, File, Response
# # from fastapi.responses import HTMLResponse
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # import os
# # import numpy as np
# # import cv2
# # from typing import Dict, Any
# #
# # from .detector import FeatureDetector
# # from .schemas import DetectResponse, StylePrediction, Detection
# # from .clarifai_client import classify_style   # NEW
# #
# # import io
# #
# # app = FastAPI(title="Instant Property Feature Detector", version="1.0.0")
# #
# # # Allow local dev origins (adjust as needed)
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# #
# # BASE_DIR = Path(__file__).resolve().parent  # backend/
# # STATIC_DIR = BASE_DIR / "static"            # backend/static/
# #
# # app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# #
# # detector = FeatureDetector()
# #
# # @app.get("/", response_class=HTMLResponse)
# # async def index():
# #     index_file = STATIC_DIR / "index.html"
# #     with open(index_file, "r", encoding="utf-8") as f:
# #         return HTMLResponse(f.read())
# #
# # @app.post("/detect", response_model=DetectResponse)
# # async def detect(file: UploadFile = File(...)) -> DetectResponse:
# #     data = await file.read()
# #     file_bytes = np.frombuffer(data, np.uint8)
# #     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# #     if img is None:
# #         raise ValueError("Invalid image")
# #
# #     # Run YOLO detector
# #     result: Dict[str, Any] = detector.predict(img)
# #
# #     # Run Clarifai style classifier
# #     styles = classify_style(img)
# #
# #     # Convert to pydantic models
# #     detections = [
# #         Detection(
# #             label=d["label"],
# #             confidence=d["confidence"],
# #             box=tuple(d["box"]),
# #             area=d["area"]
# #         )
# #         for d in result["detections"]
# #     ]
# #
# #     return DetectResponse(
# #         width=result["width"],
# #         height=result["height"],
# #         detections=detections,
# #         counts=result["counts"],
# #         amenities_score=result["amenities_score"],
# #         styles=styles   # NEW
# #     )
#
# import os
# import requests
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from .detector import FeatureDetector
#
# app = FastAPI()
#
# # Enable CORS so frontend can connect
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
#
# # Serve static frontend
# app.mount("/static", StaticFiles(directory="frontend"), name="static")
#
# @app.get("/", response_class=HTMLResponse)
# async def index():
#     with open("frontend/index.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(f.read())
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Initialize YOLO detector
# detector = FeatureDetector("yolov8n.pt")
#
# # eBay API credentials (set in .env or environment variables)
# EBAY_APP_ID = os.getenv("EBAY_APP_ID")
#
# def search_ebay(query: str, limit: int = 5):
#     """
#     Search eBay for products by keyword.
#     """
#     url = "https://svcs.ebay.com/services/search/FindingService/v1"
#     params = {
#         "OPERATION-NAME": "findItemsByKeywords",
#         "SERVICE-VERSION": "1.0.0",
#         "SECURITY-APPNAME": EBAY_APP_ID,
#         "RESPONSE-DATA-FORMAT": "JSON",
#         "REST-PAYLOAD": "",
#         "keywords": query,
#         "paginationInput.entriesPerPage": str(limit)
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#
#     items = []
#     try:
#         search_results = data["findItemsByKeywordsResponse"][0]["searchResult"][0]["item"]
#         for item in search_results:
#             items.append({
#                 "title": item["title"][0],
#                 "price": item["sellingStatus"][0]["currentPrice"][0]["__value__"],
#                 "currency": item["sellingStatus"][0]["currentPrice"][0]["@currencyId"],
#                 "url": item["viewItemURL"][0],
#                 "image": item.get("galleryURL", [""])[0]
#             })
#     except KeyError:
#         pass
#
#     return items
#
# @app.post("/detect-and-shop/")
# async def detect_and_shop(file: UploadFile = File(...)):
#     """
#     Endpoint: Upload an image → YOLO detects furniture → Search eBay.
#     """
#     # Save uploaded image
#     img_path = f"temp_{file.filename}"
#     with open(img_path, "wb") as f:
#         f.write(await file.read())
#
#     # Run YOLO detection
#     detections = detector.detect_features(img_path)
#
#     results = []
#     for det in detections:
#         label = det["label"]  # e.g. "sofa"
#         ebay_results = search_ebay(label)  # search eBay
#         results.append({
#             "object": label,
#             "confidence": det["confidence"],
#             "products": ebay_results
#         })
#
#     return JSONResponse(content={"detections": results})
#

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import numpy as np
import cv2
from .detector import FeatureDetector
from .schemas import DetectResponse, Detection

app = FastAPI(title="HomeLens Property Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("backend/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

detector = FeatureDetector()

@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    data = await file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    result: Dict[str, Any] = detector.predict(img)

    detections = [
        Detection(label=d["label"], confidence=d["confidence"], box=tuple(d["box"]), area=d["area"])
        for d in result["detections"]
    ]

    # Use detected labels for eBay search; for now, just first label or "sofa" as fallback
    query = detections[0].label if detections else "sofa"
    recommendations = detector.search_ebay(query)

    return DetectResponse(
        width=result["width"],
        height=result["height"],
        detections=detections,
        counts=result["counts"],
        amenities_score=result["amenities_score"],
        recommendations=recommendations
    )
