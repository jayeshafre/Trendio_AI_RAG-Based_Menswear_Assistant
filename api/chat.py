from fastapi import APIRouter, Request
from pydantic import BaseModel
from services.rag import ask

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(request: Request, body: ChatRequest):
    result = ask(
        user_query=body.message,
        index=request.app.state.index,
        products=request.app.state.products
    )
    return result