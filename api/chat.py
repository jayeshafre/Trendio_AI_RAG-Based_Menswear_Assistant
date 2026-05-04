from fastapi import APIRouter
from pydantic import BaseModel
from services.rag import ask

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(request: ChatRequest):
    result = ask(request.message)
    return result