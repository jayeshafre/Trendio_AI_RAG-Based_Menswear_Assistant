import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from api.search import router as search_router
from api.chat import router as chat_router
from api.recommendations import router as recommendations_router
from api.outfit import router as outfit_router
from services.vector_store import load_index
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading FAISS index into memory...")
    index, products, embeddings = load_index()
    app.state.index = index
    app.state.products = products
    app.state.embeddings = embeddings
    print(f"Loaded {len(products)} products into memory. Ready.")

    yield

    print("Shutting down Trendio AI.")


app = FastAPI(
    title="Trendio AI",
    version="1.0.0",
    description="AI-powered menswear assistant for Trendio",
    lifespan=lifespan
)

# CORS_ORIGINS env var: comma-separated list, e.g.
#   https://trendio-ecommerce-fashion-store.vercel.app,http://localhost:5173
cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(recommendations_router)
app.include_router(outfit_router)


@app.get("/")
def root():
    return {"status": "Trendio AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}