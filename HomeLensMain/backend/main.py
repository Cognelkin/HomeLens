# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from detector import detect_furniture
from clarifai import classify_style
from schemas import Detection, DetectionResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", response_model=DetectionResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detections = detect_furniture(img)

    output = []
    for det in detections:
        style = classify_style(det["crop"])
        output.append(Detection(item=det["label"], style=style))

    return DetectionResponse(detections=output)
