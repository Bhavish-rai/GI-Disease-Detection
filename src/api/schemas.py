
from typing import Literal

from pydantic import BaseModel, Field


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


class ExplanationRequest(BaseModel):
    predicted_class: str = Field(min_length=1, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=100)


class ExplanationResponse(BaseModel):
    predicted_class: str
    explanation: str
    notice: str = (
        "Educational information only. This model prediction is not "
        "a confirmed diagnosis. Consult a qualified healthcare professional."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1500)


class ChatRequest(BaseModel):
    predicted_class: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=8,
    )


class ChatResponse(BaseModel):
    answer: str
    notice: str = (
        "Educational information only; not a diagnosis or "
        "personalized treatment advice."
    )
