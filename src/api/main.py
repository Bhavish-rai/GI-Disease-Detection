from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, PredictionResponse
from src.api.utils import save_uploaded_image, delete_temp_file


app = FastAPI(
    title="GI Disease Detection API",
    description=(
        "Backend API for AI-assisted gastrointestinal disease "
        "classification from endoscopic images."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        service="GI Disease Detection API",
    )


# ---------------------------------------------------------
# Basic Image Upload Test
# ---------------------------------------------------------

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Test endpoint for image upload.

    This endpoint does not perform ML prediction yet.
    It only verifies that the backend can receive an image.
    """

    file_path = await save_uploaded_image(file)

    try:
        return {
            "success": True,
            "filename": file.filename,
            "message": "Image uploaded successfully.",
        }

    finally:
        delete_temp_file(file_path)


# ---------------------------------------------------------
# Prediction Endpoint - Temporary Version
# ---------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.

    The actual EfficientNetB0 inference will be connected here
    after the trained model and inference module are available.
    """

    file_path = await save_uploaded_image(file)

    try:
        # Temporary response.
        # This will be replaced by the real ML prediction.
        return PredictionResponse(
            predicted_class="pending",
            confidence=0.0,
            message=(
                "Image received successfully. "
                "ML inference will be connected next."
            ),
        )

    finally:
        delete_temp_file(file_path)