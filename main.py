from fastapi import FastAPI
from dotenv import load_dotenv
from api.search import router as search_router

load_dotenv()

app = FastAPI(
    title="Trendio AI",
    version="1.0.0",
    description="AI-powered menswear assistant for Trendio"
)

app.include_router(search_router)


@app.get("/")
def root():
    return {"status": "Trendio AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}