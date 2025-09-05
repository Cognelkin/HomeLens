from pydantic import BaseModel
from typing import List, Tuple, Dict

class Detection(BaseModel):
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
    area: float

class StylePrediction(BaseModel):
    name: str
    confidence: float

class DetectResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    counts: dict  # or Dict[str, int] if you prefer stricter typing
    amenities_score: float
    styles: List[StylePrediction]
