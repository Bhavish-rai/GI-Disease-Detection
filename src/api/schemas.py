
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: list[TopPrediction]
    message: str
    notice: str


class GradCAMResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_image: str
    message: str
    notice: str
