from pathlib import Path

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
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return path

    path = Path(__file__).parent / "test_sample.png"
    Image.new("RGB", (224, 224), color=(100, 120, 140)).save(path)
    return path


def payload(path):
    content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"file": (path.name, path.read_bytes(), content_type)}


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
        files={"file": ("broken.jpg", b"this is not a real image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_reject_mismatched_image_extension(client):
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (32, 32), color=(100, 120, 140)).save(
        buffer, format="PNG"
    )

    response = client.post(
        "/api/upload",
        files={"file": ("image.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_reject_oversized_image(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/upload",
        files={"file": ("large.jpg", oversized_content, "image/jpeg")},
    )
    assert response.status_code == 413


def test_gradcam_requires_file(client):
    response = client.post("/api/gradcam")
    assert response.status_code == 422
