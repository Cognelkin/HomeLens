# HomeLensMain/condition_classifier.py
from PIL import Image
import numpy as np
import cv2
import traceback

# Try to import transformers + torch for CLIP; if unavailable, we'll fallback to heuristic
_HAS_CLIP = True
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    _HAS_CLIP = False

class ConditionClassifier:
    def __init__(self, device=None):
        self.device = device
        if self.device is None and _HAS_CLIP:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CLIP prompts (damage vs clean) templates
        self.damage_templates = [
            "a photo of a damaged {obj}",
            "a photo of a broken {obj}",
            "a photo of a dirty {obj}",
            "a photo of an old {obj}",
            "a photo of a stained {obj}"
        ]
        self.clean_templates = [
            "a photo of a clean {obj}",
            "a photo of a new {obj}",
            "a photo of a well-maintained {obj}",
            "a photo of an undamaged {obj}"
        ]

        if _HAS_CLIP:
            try:
                # model name: openai/clip-vit-base-patch32 (works well for zero-shot)
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                self.clip_ok = True
            except Exception:
                # if anything goes wrong, fallback
                self.clip_ok = False
        else:
            self.clip_ok = False

    def predict(self, pil_image: Image.Image, object_label: str = None):
        """
        Returns (label, score) where label in {'Good','Damaged','Unknown'}
        and score is 0..1 (confidence of Damaged).
        object_label: optional detected type (e.g., "sink", "wall") used to specialize prompts.
        """
        try:
            if self.clip_ok:
                return self._predict_clip(pil_image, object_label)
        except Exception:
            # if CLIP errors, fall back
            traceback.print_exc()
        # fallback heuristic if CLIP not available or failed
        return self._predict_heuristic(pil_image)

    def _predict_clip(self, pil_image: Image.Image, object_label: str = None):
        # Build text prompts
        obj = (object_label or "object").lower()
        damage_texts = [t.format(obj=obj) for t in self.damage_templates]
        clean_texts = [t.format(obj=obj) for t in self.clean_templates]
        all_texts = damage_texts + clean_texts

        # Processor can handle a single image and multiple texts; it will broadcast the image
        inputs = self.clip_proc(text=all_texts, images=pil_image, return_tensors="pt", padding=True)
        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # logits_per_image shape: (batch_size=1, num_texts)
            logits = outputs.logits_per_image[0]  # first (and only) image
            probs = logits.softmax(dim=0).cpu().numpy()  # normalized across texts

        n_damage = len(damage_texts)
        damage_prob = float(probs[:n_damage].mean())
        clean_prob = float(probs[n_damage:].mean())

        # Score: relative damage probability
        score = damage_prob / (damage_prob + clean_prob + 1e-8)
        label = "Damaged" if damage_prob > clean_prob else "Good"
        return label, float(np.clip(score, 0.0, 1.0))

    def _predict_heuristic(self, pil_image: Image.Image):
        # Convert to BGR numpy (OpenCV style)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return "Unknown", 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness
        mean_brightness = float(gray.mean())/255.0

        # Laplacian var (texture/noise/cracks)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size

        # Color variance
        color_std = float(np.std(img)) / 128.0

        # Combine signals (weights tuned heuristically)
        damage_score = (
            ((1 - abs(0.5 - mean_brightness) * 2) * 0.15) +
            min(lap_var / 800.0, 1.0) * 0.35 +
            min(edge_density * 3.0, 1.0) * 0.35 +
            min(color_std, 1.0) * 0.15
        )
        damage_score = float(np.clip(damage_score, 0.0, 1.0))
        label = "Damaged" if damage_score > 0.5 else "Good"
        return label, damage_score
