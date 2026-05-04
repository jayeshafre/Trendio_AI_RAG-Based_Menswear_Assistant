from groq import Groq
from services.embedder import generate_embedding
from services.vector_store import search_from_state
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)


def format_products_for_context(products: list[dict]) -> str:
    lines = []
    for i, p in enumerate(products, 1):
        price = p.get('sale_price') or p.get('base_price')
        line = (
            f"{i}. {p['title']} | Brand: {p['brand']} | "
            f"Price: Rs.{int(float(price))} | "
            f"Colors: {p.get('colors', 'N/A')} | "
            f"Sizes: {p.get('sizes', 'N/A')} | "
            f"Category: {p.get('category_name', 'N/A')}"
        )
        lines.append(line)
    return "\n".join(lines)


def ask(user_query: str, index, products: list[dict], top_k: int = 5) -> dict:
    """
    Full RAG pipeline using pre-loaded index from app.state.
    No disk reads during request handling.
    """
    # Step 1 — Embed the query
    query_vector = generate_embedding(user_query)

    # Step 2 — Retrieve from memory
    retrieved_products = search_from_state(index, products, query_vector, top_k)

    # Step 3 — Format as context
    context = format_products_for_context(retrieved_products)

    # Step 4 — Build prompt and call Groq
    system_prompt = """You are a professional men's fashion stylist and shopping assistant for Trendio, 
a premium Indian menswear brand.

STRICT RULES — follow these without exception:
- ONLY recommend products explicitly listed in the context provided
- NEVER invent, assume or mention products that are not in the context
- NEVER mention prices that are not in the context
- If the catalog does not have enough products for a complete outfit, honestly say so
- Always mention the exact product name and exact price from the context
- Be conversational, warm and helpful
- Keep responses concise and actionable"""

    user_prompt = f"""Customer query: {user_query}

Available products from Trendio catalog:
{context}

Help the customer based on the above products only."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return {
        "query": user_query,
        "answer": answer,
        "products_used": retrieved_products
    }