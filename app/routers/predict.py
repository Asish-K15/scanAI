from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.dog_eye import get_dog_eye_model


router = APIRouter(
    prefix="/api",
    tags=["prediction"],
)


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
):
    # Temporary development router:
    # species = dog
    # body_area = eye
    #
    # Replace this routing logic later with the actual
    # species/body-area models.

    species = "dog"
    body_area = "eye"

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image.",
        )

    try:
        model = get_dog_eye_model()
        result = model.predict(pil_image)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dog eye inference failed: {exc}",
        ) from exc

    return {
        "species": species,
        "body_area": body_area,
        **result,
    }
