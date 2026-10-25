from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    message: str