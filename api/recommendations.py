from fastapi import APIRouter, Request, HTTPException, Depends
from services.vector_store import get_similar_products
from api.dependencies import verify_api_key

router = APIRouter(prefix="/recommendations", tags=["Recommendations"], dependencies=[Depends(verify_api_key)])


@router.get("/{product_id}")
def recommend(product_id: str, request: Request, top_k: int = 5):
    """
    Given a product ID, returns the most similar products.
    Called from the product detail page in React.
    
    Example: GET /recommendations/9e5f985a-143a-4977-a176-a3d882a64699
    """
    results = get_similar_products(
        product_id=product_id,
        index=request.app.state.index,
        products=request.app.state.products,
        embeddings=request.app.state.embeddings,
        top_k=top_k
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Product not found in AI index. Run build_index.py to sync."
        )

    return {
        "product_id": product_id,
        "total": len(results),
        "similar_products": results
    }