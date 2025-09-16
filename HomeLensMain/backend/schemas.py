# from pydantic import BaseModel
# from typing import List, Tuple, Dict, Any
#
# class Detection(BaseModel):
#     label: str
#     confidence: float
#     box: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
#     area: float
#
# class StylePrediction(BaseModel):
#     name: str
#     confidence: float
#
# class Recommendation(BaseModel):
#     product_id: str
#     score: float
#     metadata: Dict[str, Any] = {}
#
# class DetectedObjectWithRecs(BaseModel):
#     label: str
#     confidence: float
#     recommendations: List[Recommendation]
#
# class DetectResponse(BaseModel):
#     width: int
#     height: int
#     detections: List[Detection]
#     counts: dict  # or Dict[str, int] if you prefer stricter typing
#     amenities_score: float
#     styles: List[StylePrediction]
#     recommendations: List[DetectedObjectWithRecs]
#
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

class Detection(BaseModel):
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1,y1,x2,y2
    area: float

class DetectResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    counts: Dict[str, int]
    amenities_score: float
    recommendations: List[Dict[str, Any]]  # <- must be a list
