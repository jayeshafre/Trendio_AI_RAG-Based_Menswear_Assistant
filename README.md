# Trendio AI — RAG-Based Menswear Assistant

> An AI-powered fashion assistant integrated into a real-world menswear e-commerce platform. Built with Retrieval Augmented Generation (RAG), semantic search, and a local embedding pipeline — zero OpenAI costs for vector search.

---

## Screenshots

### AI Search Dropdown
Live semantic search results appear as the user types — powered by vector similarity, not keywords.

![AI Search Dropdown](assets/screenshots/search-dropdown.png)

---

### AI Search Results Page
Full results page at `/search` with product grid, category labels, and sale badges.

![AI Search Results](assets/screenshots/search-results.png)

---

### RAG Chat Assistant
Floating chat widget with AI-generated fashion advice and clickable product cards linking directly to product pages.

![Chat Widget](assets/screenshots/chat-widget.png)

---

### Similar Products
"You may also like" section on every product detail page — powered by vector similarity, not category filters.

![Similar Products](assets/screenshots/similar-products.png)

---

## What It Does

Trendio AI is a production-grade AI service that plugs into an existing Django + React e-commerce platform and adds four intelligent features:

- **Semantic Search** — Users search naturally ("formal blue shirt for office") instead of exact keywords. Results are ranked by meaning, not keyword match.
- **RAG Chatbot** — A fashion stylist assistant that answers questions using only real Trendio catalog data. No hallucination — strictly grounded in your product inventory.
- **Similar Products** — "You may also like" section on every product detail page, powered by vector similarity.
- **Outfit Generator** — Suggests complete outfits (top + bottom) for any occasion within a given budget.

---

## System Architecture

```
React Frontend (port 3000)
        │
        ├──► Django REST API (port 8000)    → Products, Cart, Orders, Auth
        │
        └──► FastAPI AI Service (port 8001) → Chatbot, Search, Recommendations, Outfits
                        │
                        ├── sentence-transformers  (local embeddings, free)
                        ├── FAISS                  (vector database, in-memory)
                        └── Groq API               (LLM — Llama 3.1, free tier)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Framework | FastAPI |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS (CPU) |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Dependency Management | Poetry |
| Existing Backend | Django + DRF |
| Existing Frontend | React + Vite + Tailwind |
| Database | PostgreSQL |

---

## Project Structure

```
trendio-ai/
├── main.py                    # FastAPI entry point, lifespan index loading
├── config.py                  # All settings and constants
├── services/
│   ├── embedder.py            # DB connection, text building, embedding generation
│   ├── vector_store.py        # FAISS save, load, search operations
│   ├── rag.py                 # Full RAG pipeline — retrieval + Groq generation
│   └── outfit.py              # Outfit generator — category-aware retrieval
├── api/
│   ├── dependencies.py        # API key authentication
│   ├── search.py              # Semantic search endpoint
│   ├── chat.py                # RAG chatbot endpoint
│   ├── recommendations.py     # Similar products endpoint
│   └── outfit.py              # Outfit generator endpoint
├── scripts/
│   └── build_index.py         # One-time script to build FAISS index from DB
├── data/
│   └── faiss_index/           # FAISS index files (gitignored)
├── assets/
│   └── screenshots/           # README screenshots
├── .env                       # Secrets (gitignored)
├── .env.example               # Environment variable template
├── pyproject.toml             # Poetry dependencies
└── poetry.toml                # Local virtualenv config
```

---

## API Endpoints

All endpoints require `X-Api-Key` header.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search/` | Semantic product search |
| `POST` | `/chat/` | RAG fashion chatbot |
| `GET` | `/recommendations/{product_id}` | Similar products by vector similarity |
| `POST` | `/outfit/` | Outfit generator by occasion and budget |
| `GET` | `/health` | Health check |

### Example Requests

**Semantic Search**
```json
POST /search/
{
  "query": "formal blue shirt for office",
  "top_k": 5
}
```

**RAG Chatbot**
```json
POST /chat/
{
  "message": "Suggest an outfit for a wedding under Rs.3000"
}
```

**Recommendations**
```
GET /recommendations/9e5f985a-143a-4977-a176-a3d882a64699?top_k=4
```

**Outfit Generator**
```json
POST /outfit/
{
  "occasion": "formal office meeting",
  "budget": 3000
}
```

---

## How RAG Works Here

```
User Query
    │
    ▼
Generate Embedding          ← sentence-transformers (local, free)
    │
    ▼
Search FAISS Index          ← finds top-K similar product vectors
    │
    ▼
Format Product Context      ← builds readable text from product data
    │
    ▼
Groq LLM (Llama 3.1)       ← generates response using ONLY retrieved products
    │
    ▼
Filter by Mention           ← only return products the LLM actually mentioned
    │
    ▼
Response + Product Cards
```

**Memory Hook:** `RAG = Brain (LLM) + Memory (your product catalog)`

---

## Frontend Features

Built into the existing Trendio React frontend:

- **AI Search Dropdown** — Live results appear as user types (400ms debounce), powered by semantic search
- **AI Search Results Page** — Full results page at `/search` with product grid
- **Chat Widget** — Floating chat button on every page, shows product cards with direct links
- **Similar Products** — "You may also like" section on product detail page

---

## Local Setup

### Prerequisites

- Python 3.11+
- Poetry 2.x
- PostgreSQL (existing Trendio database)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/jayeshafre/trendio-ai.git
cd trendio-ai

# Configure Poetry to use local virtualenv
poetry config virtualenvs.in-project true --local

# Set Python version
poetry env use python3.11

# Install dependencies
poetry install

# Create .env file
cp .env.example .env
# Fill in your values
```

### Environment Variables

```env
GROQ_API_KEY=gsk_your_key_here
TRENDIO_DATABASE_URL=postgresql://user:password@localhost:5432/trendio_db
FAISS_INDEX_PATH=data/faiss_index
AI_SECRET_KEY=your-secret-key-here
```

### Build the AI Index

Run this once after setup, and again whenever products are added in bulk:

```bash
poetry run python scripts/build_index.py
```

Expected output:
```
Step 1: Fetching products from Trendio database...
Found 19 active products.
Step 2: Generating embeddings...
  Embedded 19/19 products...
Step 3: Building and saving FAISS index...
Saved 19 products to FAISS index.
Done. FAISS index is ready.
```

### Start the Server

```bash
poetry run uvicorn main:app --reload --port 8001
```

Visit `http://localhost:8001/docs` for interactive API documentation.

---

## Key Design Decisions

**Why FastAPI over Django for AI?**
FastAPI handles async operations natively — critical when running embeddings, vector search, and LLM calls simultaneously. Django is kept for existing e-commerce logic.

**Why sentence-transformers over OpenAI embeddings?**
Runs 100% locally. Zero cost per search query. No API rate limits. The `all-MiniLM-L6-v2` model is 80MB, cached after first download, and produces 384-dimension vectors sufficient for product catalog search.

**Why FAISS over Pinecone?**
For a catalog under 100k products, FAISS running in-memory is faster than a network call to Pinecone. Index loads once at startup — every search query reads from RAM, not disk.

**Why Groq over OpenAI for LLM?**
Free tier with generous rate limits. Llama 3.1 on Groq is fast enough for real-time chat. Swapping to GPT-4 later is a one-line change in `config.py`.

**Hallucination Prevention**
The system prompt strictly forbids the LLM from mentioning products not in the retrieved context. Temperature set to 0.5 for focused, consistent responses.

---

## Author

**Jayesh Afre**
GitHub: [@jayeshafre](https://github.com/jayeshafre)

---

## License

MIT
