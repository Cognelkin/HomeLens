import os
from clarifai.client.model import Model
import cv2
from dotenv import load_dotenv

load_dotenv()
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
if not CLARIFAI_PAT:
    raise RuntimeError("Clarifai PAT not set. Please set CLARIFAI_PAT env var.")

# Clarifai Style Model
STYLE_MODEL_ID = "general-image-recognition"
#STYLE_MODEL_VERSION_ID = "aa7f35c01e0642fda5cf400f543e7c40"  # optional but safer

def classify_style(image_bgr):
    _, buf = cv2.imencode(".jpg", image_bgr)
    img_bytes = buf.tobytes()

    # Initialize model client
    model = Model(STYLE_MODEL_ID, pat=CLARIFAI_PAT)

    # Run prediction
    pred = model.predict_by_bytes(img_bytes, input_type="image")

    styles = []
    for concept in pred.outputs[0].data.concepts[:5]:  # top 5 predictions
        styles.append({
            "name": concept.name,
            "confidence": concept.value
        })
    return styles
