import base64
import cv2
import numpy as np
from clarifai.client.model import Model
from clarifai.client.input import Inputs

# Use your own dataset + embedding model
SIMILARITY_MODEL_ID = "general-image-embedding"
APP_ID = "YOUR_APP_ID"
USER_ID = "YOUR_USER_ID"

model = Model(user_id=USER_ID, app_id=APP_ID, model_id=SIMILARITY_MODEL_ID)

def find_similar_products(crop_img: np.ndarray, top_k: int = 3):
    """Send crop to Clarifai and return similar products."""
    _, buf = cv2.imencode(".jpg", crop_img)
    b64_img = base64.b64encode(buf).decode("utf-8")

    response = model.predict(
        inputs=[Inputs(image_base64=b64_img)]
    )

    results = []
    for r in response.outputs[0].data.embeddings[:top_k]:
        results.append({
            "product_id": r.id,
            "score": r.score,
            # If you stored metadata in Clarifai (url, price, etc.)
            "metadata": getattr(r, "metadata", {})
        })

    return results
