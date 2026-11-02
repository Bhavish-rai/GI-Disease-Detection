
from pathlib import Path
from unittest.mock import patch
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_image():
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "dataset", root / "data", root / "datasets"]

    for directory in candidates:
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png"
                }:
                    return path

    path = Path(__file__).parent / "test_sample.png"
    Image.new("RGB", (224, 224), color=(100, 120, 140)).save(path)
    return path


def payload(path):
    content_type = (
        "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    )
    return {"file": (path.name, path.read_bytes(), content_type)}


# --------------------------------------------------
# EXISTING API TESTS
# --------------------------------------------------

def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload(client, sample_image):
    response = client.post("/api/upload", files=payload(sample_image))
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_predict(client, sample_image):
    response = client.post("/api/predict", files=payload(sample_image))
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"]
    assert 0 <= data["confidence"] <= 100
    assert len(data["top_predictions"]) == 3


def test_gradcam(client, sample_image):
    response = client.post("/api/gradcam", files=payload(sample_image))
    assert response.status_code == 200

    data = response.json()
    image_response = client.get(data["gradcam_image"])

    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert image_response.content.startswith(b"\x89PNG")


def test_reject_unsupported_file(client):
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_requires_file(client):
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_reject_corrupted_image(client):
    response = client.post(
        "/api/upload",
        files={
            "file": (
                "broken.jpg",
                b"this is not a real image",
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 400


def test_reject_mismatched_image_extension(client):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 120, 140)).save(
        buffer, format="PNG"
    )

    response = client.post(
        "/api/upload",
        files={
            "file": (
                "image.jpg",
                buffer.getvalue(),
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 400


def test_reject_oversized_image(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/upload",
        files={
            "file": (
                "large.jpg",
                oversized_content,
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 413


def test_gradcam_requires_file(client):
    response = client.post("/api/gradcam")
    assert response.status_code == 422


# --------------------------------------------------
# GEMINI EXPLANATION TESTS
# --------------------------------------------------

def test_gemini_explanation_success(client):
    expected_text = (
        "What This Class Means\n"
        "This is educational information about polyps."
    )

    with patch(
        "src.api.main.generate_explanation",
        return_value=expected_text,
    ) as mocked:
        response = client.post(
            "/api/explanation",
            json={
                "predicted_class": "polyps",
                "confidence": 92.5,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "polyps"
    assert data["explanation"] == expected_text
    assert "not" in data["notice"].lower()
    mocked.assert_called_once_with(
        predicted_class="polyps",
        confidence=92.5,
    )


def test_gemini_explanation_rejects_invalid_class(client):
    with patch("src.api.main.generate_explanation") as mocked:
        mocked.side_effect = ValueError("Unsupported predicted class.")

        response = client.post(
            "/api/explanation",
            json={
                "predicted_class": "unknown-disease",
                "confidence": 90,
            },
        )

    assert response.status_code == 400
    mocked.assert_called_once()


def test_gemini_explanation_missing_key(client):
    with patch(
        "src.api.main.generate_explanation",
        side_effect=RuntimeError(
            "GEMINI_API_KEY is missing. Configure it in the project root .env file."
        ),
    ):
        response = client.post(
            "/api/explanation",
            json={"predicted_class": "polyps"},
        )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_gemini_chat_success(client):
    expected_answer = "Polyps are growths that can occur in the digestive tract."

    with patch(
        "src.api.main.answer_question",
        return_value=expected_answer,
    ) as mocked:
        response = client.post(
            "/api/explanation/chat",
            json={
                "predicted_class": "polyps",
                "question": "What are polyps?",
                "history": [
                    {"role": "user", "content": "Explain the prediction."},
                    {"role": "assistant", "content": "The class is polyps."},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == expected_answer
    mocked.assert_called_once_with(
        predicted_class="polyps",
        question="What are polyps?",
        history=[
            {"role": "user", "content": "Explain the prediction."},
            {"role": "assistant", "content": "The class is polyps."},
        ],
    )


def test_gemini_chat_rejects_invalid_class(client):
    with patch(
        "src.api.main.answer_question",
        side_effect=ValueError("Unsupported predicted class."),
    ):
        response = client.post(
            "/api/explanation/chat",
            json={
                "predicted_class": "unknown-disease",
                "question": "What does this mean?",
            },
        )

    assert response.status_code == 400


def test_gemini_chat_rejects_empty_question(client):
    with patch("src.api.main.answer_question") as mocked:
        response = client.post(
            "/api/explanation/chat",
            json={
                "predicted_class": "polyps",
                "question": "",
            },
        )

    assert response.status_code == 422
    mocked.assert_not_called()


def test_gemini_chat_handles_service_unavailable(client):
    with patch(
        "src.api.main.answer_question",
        side_effect=RuntimeError("Gemini service unavailable."),
    ):
        response = client.post(
            "/api/explanation/chat",
            json={
                "predicted_class": "polyps",
                "question": "What does this class mean?",
            },
        )

    assert response.status_code == 503
