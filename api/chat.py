from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from services.rag import ask
from api.dependencies import verify_api_key

router = APIRouter(prefix="/chat", tags=["Chat"], dependencies=[Depends(verify_api_key)])


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