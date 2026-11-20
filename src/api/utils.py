
from io import BytesIO
from pathlib import Path
import uuid

from fastapi import UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

EXPECTED_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
}

TEMP_DIR = Path("reports") / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_image(file: UploadFile) -> Path:
    """Validate and temporarily save an uploaded image."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: {extension}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    try:
        # Read at most 10 MB plus one byte to detect oversized uploads.
        contents = await file.read(MAX_UPLOAD_SIZE + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the maximum upload size of 10 MB.",
            )

        # Verify the actual image data, not just the filename.
        with Image.open(BytesIO(contents)) as image:
            if image.format != EXPECTED_FORMATS[extension]:
                raise HTTPException(
                    status_code=400,
                    detail="File extension does not match the image format.",
                )

            image.verify()

        # Save only after validation succeeds.
        unique_name = f"{uuid.uuid4().hex}{extension}"
        file_path = TEMP_DIR / unique_name
        file_path.write_bytes(contents)

        return file_path

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded image.",
        ) from exc


def delete_temp_file(file_path: Path) -> None:
    """Delete a temporary uploaded file."""

    try:
        if file_path.exists():
            file_path.unlink()
    except OSError:
        pass
