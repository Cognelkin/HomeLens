# Instant Property Feature Detector

Detect common real-estate-relevant features (sofas, beds, TV, sink, toilet, refrigerator, etc.) in photos using a **pretrained YOLOv8** model—no training required. Includes a **FastAPI** backend and a **vanilla JS** single-page frontend that draws bounding boxes, tallies features, and computes a simple amenities score.

https://github.com/ultralytics/ultralytics (YOLOv8) is used via pip.

## Quick Start

1) Create and activate a Python 3.9+ virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the server:

```bash
uvicorn HomeLensMain.main:HomeLensMain --reload
```

4) Open the app:

- Visit: **http://127.0.0.1:8000/**

5) Try it out:
- Upload a room/property photo (JPEG/PNG).
- The app will display detections, counts per feature, and a simple amenities score (0–100).

## What it detects (from COCO classes)

The included model (YOLOv8n by default) can detect, among many classes, the following features that are often useful for property photos:

- bed, couch (sofa), chair, dining table
- tv
- sink, toilet, refrigerator, oven, microwave
- potted plant (decor), vase
- laptop (useful if detecting "workspace" areas)
- bathtub is not a COCO class; "toilet" often helps identify bathrooms
- window/door are not available in COCO default; you can later swap in a model trained on those if needed.

The UI filters detections to a curated set of **property_features**. You can tweak this list in `app/detector.py`.

## Tech Stack

- **FastAPI** for the backend API and static file serving
- **Ultralytics YOLOv8** (pretrained) for detection
- **OpenCV** for basic image handling
- **Vanilla JS** for a minimal, dependency-free frontend that draws boxes and shows analytics

## Project Structure

```
property-feature-detector/
├─ app/
│  ├─ main.py          # FastAPI app & routes
│  ├─ detector.py      # YOLO model loader & predict logic
│  ├─ schemas.py       # Pydantic response models
│  └─ static/
│     ├─ index.html    # single-page front-end
│     ├─ style.css
│     └─ app.js
├─ requirements.txt
└─ README.md
```

## Notes

- By default this uses `yolov8n` (nano) weights for speed. You can switch to larger models
  like `yolov8s` or `yolov8m` in `app/detector.py` for higher accuracy (at some cost to latency).
- If you run on a GPU machine with CUDA/cuDNN installed, PyTorch will automatically accelerate inference.
- The **amenities score** is a simple heuristic—feel free to tailor the weights or add your own rule-based logic.
