# backend/clarifai.py
import os
from clarifai.client.model import Model
import cv2
from dotenv import load_dotenv

load_dotenv()

# Load Clarifai model (style recognition)
clarifai_model = Model(
    os.getenv("CLARIFAI_MODEL_URL"),
    pat=os.getenv("CLARIFAI_PAT")
)

def classify_style(crop):
    """
    Takes a cropped furniture image (numpy array) and classifies style.
    """
    _, buffer = cv2.imencode(".jpg", crop)
    crop_bytes = buffer.tobytes()

    response = clarifai_model.predict_by_bytes(crop_bytes, input_type="image")
    if response.outputs and response.outputs[0].data.concepts:
        return response.outputs[0].data.concepts[0].name
    return "Unknown"
