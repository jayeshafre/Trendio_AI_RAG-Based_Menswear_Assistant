"""
Run this once to build the FAISS index from your Trendio products.
Run again whenever new products are added in bulk.

Usage:
    poetry run python scripts/build_index.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedder import get_products_from_db, build_product_text, generate_embedding
from services.vector_store import save_index


def main():
    print("Step 1: Fetching products from Trendio database...")
    products = get_products_from_db()
    print(f"Found {len(products)} active products.")

    if not products:
        print("No products found. Add products in Django admin first.")
        return

    print("Step 2: Generating embeddings...")
    embeddings = []

    for i, product in enumerate(products):
        text = build_product_text(product)
        embedding = generate_embedding(text)
        embeddings.append(embedding)

        if (i + 1) % 5 == 0 or (i + 1) == len(products):
            print(f"  Embedded {i + 1}/{len(products)} products...")

    print("Step 3: Building and saving FAISS index...")
    save_index(embeddings, products)

    print("\nDone. FAISS index is ready.")


if __name__ == "__main__":
    main()