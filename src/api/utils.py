from pathlib import Path
import uuid

from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEMP_DIR = Path("reports") / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_image(file: UploadFile) -> Path:
    """
    Validate and temporarily save an uploaded image.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
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

    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = TEMP_DIR / unique_name

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        file_path.write_bytes(contents)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded image: {exc}",
        ) from exc

    return file_path


def delete_temp_file(file_path: Path) -> None:
    """
    Delete a temporary uploaded file.
    """

    try:
        if file_path.exists():
            file_path.unlink()
    except OSError:
        pass