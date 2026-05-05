import os
from dotenv import load_dotenv

load_dotenv()

AI_SECRET_KEY = os.getenv("AI_SECRET_KEY")

# Groq — for LLM responses
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Trendio Postgres database — read only
TRENDIO_DATABASE_URL = os.getenv("TRENDIO_DATABASE_URL")

# FAISS index storage path
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

# Embedding model — runs locally, no API calls, no cost
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Groq LLM model
LLM_MODEL = "llama-3.1-8b-instant"
