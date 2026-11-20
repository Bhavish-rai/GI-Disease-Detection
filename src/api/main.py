
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.schemas import (
    HealthResponse,
    PredictionResponse,
    GradCAMResponse,
    ExplanationRequest,
    ExplanationResponse,
    ChatRequest,
    ChatResponse,
)
from src.api.utils import save_uploaded_image, delete_temp_file
from src.inference.predict import load_trained_model, predict_image
from src.inference.gradcam import generate_gradcam
from src.ai.gemini_explainer import (
    generate_explanation,
    answer_question,
)


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
        "API for gastrointestinal endoscopy image classification, "
        "Grad-CAM visualization, and Gemini-powered educational "
        "explanations and follow-up questions."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Check whether the API is running."""
    return HealthResponse(
        status="healthy",
        service="GI Disease Detection API",
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

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


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded gastrointestinal endoscopy image."""
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


# --------------------------------------------------
# GRAD-CAM VISUALIZATION
# --------------------------------------------------

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


# --------------------------------------------------
# GEMINI AI EXPLANATION
# --------------------------------------------------

@app.post(
    "/api/explanation",
    response_model=ExplanationResponse,
)
def explain_prediction(request: ExplanationRequest):
    """Generate educational information about a predicted class."""
    try:
        explanation = generate_explanation(
            predicted_class=request.predicted_class,
            confidence=request.confidence,
        )

        return ExplanationResponse(
            predicted_class=request.predicted_class.strip().lower(),
            explanation=explanation,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        # For example, a missing Gemini API key or empty response.
        logger.warning("Gemini explanation unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to generate AI explanation.")
        raise HTTPException(
            status_code=502,
            detail="The AI explanation service is temporarily unavailable.",
        ) from exc


# --------------------------------------------------
# GEMINI AI FOLLOW-UP CHAT
# --------------------------------------------------

@app.post(
    "/api/explanation/chat",
    response_model=ChatResponse,
)
def chat_about_prediction(request: ChatRequest):
    """Answer a follow-up question about the predicted class."""
    try:
        answer = answer_question(
            predicted_class=request.predicted_class,
            question=request.question,
            history=[
                message.model_dump()
                for message in request.history
            ],
        )

        return ChatResponse(answer=answer)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.warning("Gemini chat unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to answer AI follow-up question.")
        raise HTTPException(
            status_code=502,
            detail="The AI chat service is temporarily unavailable.",
        ) from exc
