from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.body_area_router import route_body_area
from app.services.dog_eye import get_dog_eye_model
from app.services.recommendation import build_recommendation
from app.services.species_router import route_species


router = APIRouter(
    prefix="/api",
    tags=["prediction"],
)


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    species: str = Form(...),
    body_area: str = Form(...),
):
    """
    Route an uploaded animal image to the appropriate model.

    Current implementation:
    - Species and body area are explicitly supplied by the client.
    - Dog + eye routes to the frozen dog-eye-v1 model.
    - The model output is transformed into the approved v1
      recommendation schema.
    - Other model routes are not implemented yet.
    """

    try:
        routed_species = route_species(species)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    try:
        routed_body_area = route_body_area(body_area)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

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

    # --------------------------------------------------------
    # Model routing
    # --------------------------------------------------------

    if routed_species == "dog" and routed_body_area == "eye":

        try:
            model = get_dog_eye_model()
            result = model.predict(pil_image)

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Dog eye inference failed: {exc}",
            ) from exc

        return build_recommendation(
            routed_species,
            routed_body_area,
            result,
        )

    # --------------------------------------------------------
    # Models not implemented yet
    # --------------------------------------------------------

    raise HTTPException(
        status_code=501,
        detail=(
            f"No model route is currently implemented for "
            f"species='{routed_species}', "
            f"body_area='{routed_body_area}'."
        ),
    )