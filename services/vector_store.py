import os
import json
import numpy as np
import faiss
from config import FAISS_INDEX_PATH


def save_index(embeddings: list[list[float]], products: list[dict]):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    vectors = np.array(embeddings, dtype="float32")
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
    Loads FAISS index, product metadata and embeddings from disk.
    Called once at startup — everything goes into app.state.
    """
    index_path = f"{FAISS_INDEX_PATH}/index.faiss"
    products_path = f"{FAISS_INDEX_PATH}/products.json"
    embeddings_path = f"{FAISS_INDEX_PATH}/embeddings.npy"

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS index not found. Run scripts/build_index.py first."
        )

    index = faiss.read_index(index_path)

    with open(products_path, "r") as f:
        products = json.load(f)

    embeddings = np.load(embeddings_path)

    return index, products, embeddings


def search_from_state(index, products: list[dict], query_vector: list[float], top_k: int = 5) -> list[dict]:
    query = np.array([query_vector], dtype="float32")
    D, I = index.search(query, top_k)

    results = []
    for i, idx in enumerate(I[0]):
        if idx != -1:
            product = products[idx].copy()
            product['relevance_score'] = float(D[0][i])
            results.append(product)

    return results


def get_similar_products(
    product_id: str,
    index,
    products: list[dict],
    embeddings: np.ndarray,
    top_k: int = 5
) -> list[dict]:
    """
    Given a product ID, finds the most similar products using
    that product's stored embedding vector.
    No re-embedding needed — reads directly from memory.
    """
    # Find the position of this product in our products list
    product_position = None
    for i, p in enumerate(products):
        if str(p['id']) == str(product_id):
            product_position = i
            break

    if product_position is None:
        return []

    # Get that product's embedding vector
    product_vector = embeddings[product_position].reshape(1, -1)

    # Search for similar — top_k + 1 because the product itself will appear
    D, I = index.search(product_vector, top_k + 1)

    results = []
    for i, idx in enumerate(I[0]):
        if idx != -1:
            similar_product = products[idx].copy()
            # Exclude the product itself from results
            if str(similar_product['id']) == str(product_id):
                continue
            similar_product['relevance_score'] = float(D[0][i])
            results.append(similar_product)

    # Return exactly top_k results
    return results[:top_k]