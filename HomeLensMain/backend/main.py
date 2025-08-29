from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from backend.detector import detect_objects
from backend.clarifai_client import classify_and_recommend

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Step 1: Detect furniture with YOLO
    detections = detect_objects(img)

    # Step 2: Send whole image (or crops) to Clarifai
    recommendations = classify_and_recommend(image_bytes)

    return {
        "detections": detections,
        "recommendations": recommendations
    }