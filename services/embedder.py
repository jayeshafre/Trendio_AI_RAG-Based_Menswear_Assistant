import psycopg2
from sentence_transformers import SentenceTransformer
from config import TRENDIO_DATABASE_URL, EMBEDDING_MODEL

# Loads the model into memory once when the file is first imported.
# First run downloads ~80MB model and caches it permanently.
model = SentenceTransformer(EMBEDDING_MODEL)


def get_products_from_db() -> list[dict]:
    """
    Reads all active products from Trendio Postgres.
    Read-only — never writes anything to Django database.
    """
    conn = psycopg2.connect(TRENDIO_DATABASE_URL)
    cur = conn.cursor()

    cur.execute("""
        WITH RECURSIVE category_path AS (
            SELECT 
                id,
                name,
                parent_id,
                name::text AS full_path
            FROM categories
            WHERE parent_id IS NULL

            UNION ALL

            SELECT 
                c.id,
                c.name,
                c.parent_id,
                cp.full_path || ' > ' || c.name
            FROM categories c
            JOIN category_path cp ON c.parent_id = cp.id
        )
        SELECT 
            p.id,
            p.title,
            p.description,
            p.brand,
            p.slug,
            p.base_price,
            p.sale_price,
            cp.full_path as category_name,
            STRING_AGG(DISTINCT pv.color, ', ') as colors,
            STRING_AGG(DISTINCT pv.size, ', ')  as sizes,
            MIN(pi.image) as primary_image
        FROM products p
        LEFT JOIN category_path cp ON p.category_id = cp.id
        LEFT JOIN product_variants pv 
               ON pv.product_id = p.id AND pv.is_active = true
        LEFT JOIN product_images pi 
               ON pi.product_id = p.id AND pi.is_primary = true
        WHERE p.is_active = true
        GROUP BY p.id, p.title, p.description, p.brand,p.slug,
                 p.base_price, p.sale_price, cp.full_path
        ORDER BY p.created_at DESC
    """)

    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def build_product_text(product: dict) -> str:
    """
    Converts one product dict into a rich text string for embedding.
    The quality of this text directly determines search quality.
    """
    price = product['sale_price'] or product['base_price']

    parts = [
        f"{product['title']}.",
        f"Brand: {product['brand']}." if product['brand'] else "",
        f"Category: {product['category_name']}." if product['category_name'] else "",
        f"Colors available: {product['colors']}." if product['colors'] else "",
        f"Sizes available: {product['sizes']}." if product['sizes'] else "",
        f"Price: Rs.{int(price)}." if price else "",
        product['description'],
    ]

    return " ".join(p for p in parts if p.strip())


def generate_embedding(text: str) -> list[float]:
    """
    Converts text to a 384-dimension vector using local model.
    Zero API calls. Zero cost. Runs on your CPU.
    """
    embedding = model.encode(text)
    return embedding.tolist()