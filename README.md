**HomeLens**

A lightweight web application that detects real-estate-relevant features (beds, couches, TVs, sinks, toilets, refrigerators, etc.) from property photos using a pretrained YOLOv8 model.

The project includes a FastAPI backend and a minimal vanilla JavaScript frontend that renders bounding boxes, tallies detected amenities, and calculates a simple amenities score.

No model training required.

**Overview**

This app allows users to upload a property or room image and automatically identify common household features using object detection.

The backend performs inference using a pretrained YOLOv8 model, and the frontend visualizes results in real time using the HTML5 Canvas API.

The goal of this project is to demonstrate:

- Applied computer vision with pretrained models

- Backend API design with FastAPI

- Lightweight rule-based scoring logic on top of ML outputs

**Tech Stack**

Backend

- FastAPI

- Ultralytics YOLOv8 (pretrained, COCO dataset)

- PyTorch (CPU/GPU inference)

- OpenCV (image decoding and preprocessing)

- Pydantic (response schemas)

Frontend

- JavaScript

- HTML5 Canvas API (bounding box rendering)

- Minimal CSS (no frameworks)

YOLOv8 is installed via pip from:
https://github.com/ultralytics/ultralytics

**How It Works**

1. User uploads a JPEG or PNG image.

2. The FastAPI backend:

 - Loads the YOLOv8 model

 - Runs object detection

 - Filters detections to a curated list of property-related features

3. The frontend:

 - Draws bounding boxes on a canvas

 - Displays counts per feature

 - Computes an amenities score (0–100)

 - Inference runs locally. No external vision APIs are used.

**Quick Start**

1. Create virtual environment: python -m venv .venv
2. Activate virtual environment: .venv\Scripts\activate (Windows) and .venv\Scripts\activate (mac/linux)
3. Install dependencies: pip install -r requirements.txt
4. Start server: uvicorn HomeLensMain.main:HomeLensMain --reload
5. Visit http://127.0.0.1:8000/
