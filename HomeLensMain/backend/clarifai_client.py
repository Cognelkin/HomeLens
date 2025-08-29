import os
from clarifai.client.model import Model
from clarifai.client.input import Image as ClarifaiImage
from dotenv import load_dotenv

load_dotenv()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
if not CLARIFAI_PAT:
    raise ValueError("Clarifai API key not found. Please set CLARIFAI_API_KEY in .env")

# Load general or custom-trained model (Clarifai has furniture-specific models too)
model = Model("https://clarifai.com/clarifai/main/models/general-image-recognition", api_key=CLARIFAI_PAT)


def classify_and_recommend(image_bytes: bytes):
    """Send an image to Clarifai for classification + recommendations."""
    try:
        clarifai_img = ClarifaiImage(base64=image_bytes)
        response = model.predict(inputs=[clarifai_img])

        if not response.outputs:
            return {"error": "No classification results"}

        # Extract top predictions
        concepts = response.outputs[0].data.concepts
        results = [{"name": c.name, "confidence": c.value} for c in concepts[:5]]

        # Placeholder: Later we can extend this to use Clarifai's similarity search
        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}