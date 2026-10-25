
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.schemas import (
    HealthResponse,
    PredictionResponse,
    GradCAMResponse,
)
from src.api.utils import save_uploaded_image, delete_temp_file
from src.inference.predict import load_trained_model, predict_image
from src.inference.gradcam import generate_gradcam


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRADCAM_DIR = PROJECT_ROOT / "reports" / "gradcam"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the API starts."""
    try:
        app.state.model = load_trained_model()
        logger.info("EfficientNetB0 model loaded successfully.")
    except Exception:
        logger.exception("Could not load the trained model.")
        raise

    yield

    app.state.model = None


app = FastAPI(
    title="GI Disease Detection API",
    description=(
        "Backend API for AI-assisted gastrointestinal disease "
        "classification from endoscopic images."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        service="GI Disease Detection API",
    )


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Validate that the uploaded file is a supported image."""
    file_path = await save_uploaded_image(file)

    try:
        return {
            "success": True,
            "filename": file.filename,
            "message": "Image uploaded and validated successfully.",
        }
    finally:
        delete_temp_file(file_path)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded gastrointestinal image."""
    file_path = await save_uploaded_image(file)

    try:
        result = predict_image(
            file_path,
            model=app.state.model,
        )

        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            message="Prediction completed successfully.",
            notice=result["notice"],
        )

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Image prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from exc

    finally:
        delete_temp_file(file_path)


@app.post("/api/gradcam", response_model=GradCAMResponse)
async def gradcam(file: UploadFile = File(...)):
    """Generate a Grad-CAM visualization for an uploaded image."""
    file_path = await save_uploaded_image(file)

    try:
        output_path = GRADCAM_DIR / f"{uuid.uuid4().hex}_gradcam.png"

        result = generate_gradcam(
            file_path,
            model=app.state.model,
            output_path=output_path,
        )

        return GradCAMResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            gradcam_image=f"/api/gradcam/image/{output_path.name}",
            message="Grad-CAM visualization generated successfully.",
            notice=result["notice"],
        )

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the Grad-CAM visualization.",
        ) from exc

    finally:
        delete_temp_file(file_path)


@app.get("/api/gradcam/image/{filename}")
def get_gradcam_image(filename: str):
    """Serve a generated Grad-CAM PNG image."""
    if (
        Path(filename).name != filename
        or not filename.endswith("_gradcam.png")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid Grad-CAM filename.",
        )

    image_path = GRADCAM_DIR / filename

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image not found.",
        )

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=filename,
    )
