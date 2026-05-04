from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from api.search import router as search_router
from api.chat import router as chat_router
from services.vector_store import load_index

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup — loads FAISS index into memory.
    Every request reads from app.state instead of disk.
    """
    print("Loading FAISS index into memory...")
    index, products = load_index()
    app.state.index = index
    app.state.products = products
    print(f"Loaded {len(products)} products into memory. Ready.")

    yield  # server runs here

    # anything after yield runs on shutdown
    print("Shutting down Trendio AI.")


app = FastAPI(
    title="Trendio AI",
    version="1.0.0",
    description="AI-powered menswear assistant for Trendio",
    lifespan=lifespan
)

app.include_router(search_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"status": "Trendio AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}