from fastapi import APIRouter
from pydantic import BaseModel
from services.embedder import generate_embedding
from services.vector_store import search

router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/")
def semantic_search(request: SearchRequest):
    query_vector = generate_embedding(request.query)
    results = search(query_vector, top_k=request.top_k)

    return {
        "query": request.query,
        "total": len(results),
        "results": results
    }