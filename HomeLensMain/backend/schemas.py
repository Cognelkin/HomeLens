# from pydantic import BaseModel
# from typing import List, Tuple
#
# class Detection(BaseModel):
#     label: str
#     confidence: float
#     box: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
#     area: float
#
# class DetectResponse(BaseModel):
#     width: int
#     height: int
#     detections: List[Detection]
#     counts: dict
#     amenities_score: float

from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    item: str
    style: str

class DetectionResponse(BaseModel):
    detections: List[Detection]