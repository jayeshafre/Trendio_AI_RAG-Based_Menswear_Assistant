from fastapi import APIRouter, Request
from pydantic import BaseModel
from services.embedder import generate_embedding
from services.vector_store import search_from_state

router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/")
def semantic_search(request: Request, body: SearchRequest):
    query_vector = generate_embedding(body.query)
    results = search_from_state(
        request.app.state.index,
        request.app.state.products,
        query_vector,
        top_k=body.top_k
    )
    return {
        "query": body.query,
        "total": len(results),
        "results": results
    }