from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from services.outfit import generate_outfit

router = APIRouter(prefix="/outfit", tags=["Outfit Generator"])


class OutfitRequest(BaseModel):
    occasion: str
    budget: float


@router.post("/")
def outfit_generator(request: Request, body: OutfitRequest):
    if body.budget <= 0:
        raise HTTPException(
            status_code=400,
            detail="Budget must be greater than 0."
        )

    result = generate_outfit(
        occasion=body.occasion,
        budget=body.budget,
        index=request.app.state.index,
        products=request.app.state.products,
        embeddings=request.app.state.embeddings
    )

    return result