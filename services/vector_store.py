import os
import json
import numpy as np
import faiss
from config import FAISS_INDEX_PATH


def save_index(embeddings: list[list[float]], products: list[dict]):
    """
    Saves vectors + product metadata to disk.
    Creates two files:
      index.faiss    → the math (for searching)
      products.json  → the data (for retrieving after a match)
    """
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    vectors = np.array(embeddings, dtype="float32")

    # Dimension detected automatically — 384 for our model
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, f"{FAISS_INDEX_PATH}/index.faiss")

    with open(f"{FAISS_INDEX_PATH}/products.json", "w") as f:
        json.dump(products, f, default=str)

    print(f"Saved {index.ntotal} products to FAISS index.")
    print(f"Vector dimension: {dimension}")


def load_index():
    """
    Loads FAISS index and product metadata from disk.
    Called once when FastAPI starts.
    """
    index_path = f"{FAISS_INDEX_PATH}/index.faiss"
    products_path = f"{FAISS_INDEX_PATH}/products.json"

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS index not found. Run scripts/build_index.py first."
        )

    index = faiss.read_index(index_path)

    with open(products_path, "r") as f:
        products = json.load(f)

    return index, products


def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Finds the top_k most similar products to the query vector.
    Runs in milliseconds regardless of catalog size.
    """
    index, products = load_index()

    query = np.array([query_vector], dtype="float32")

    # D = distances, I = indices of matching products
    D, I = index.search(query, top_k)

    results = []
    for i, idx in enumerate(I[0]):
        if idx != -1:
            product = products[idx].copy()
            # Lower distance = more similar
            product['relevance_score'] = float(D[0][i])
            results.append(product)

    return results