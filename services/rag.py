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
            f"Category: {p.get('category_name', 'N/A')} | "
            f"Price: Rs.{int(float(price))} | "
            f"Colors: {p.get('colors', 'N/A')} | "
            f"Sizes: {p.get('sizes', 'N/A')}"
        )
        lines.append(line)
    return "\n".join(lines)


def filter_products_by_mention(answer: str, products: list[dict]) -> list[dict]:
    """
    Returns only products whose title appears in the LLM answer.
    Prevents showing irrelevant context products as cards.
    """
    mentioned = []
    answer_lower = answer.lower()
    for p in products:
        if p['title'].lower() in answer_lower:
            mentioned.append(p)
    return mentioned


def ask(user_query: str, index, products: list[dict], top_k: int = 5) -> dict:
    # Step 1 — Embed the query
    query_vector = generate_embedding(user_query)

    # Step 2 — Retrieve relevant products
    retrieved_products = search_from_state(index, products, query_vector, top_k=top_k)

    # Step 3 — Format as context
    context = format_products_for_context(retrieved_products)

    # Step 4 — Call Groq with strict filtering instructions
    system_prompt = """You are a professional men's fashion stylist and shopping assistant for Trendio,
a premium Indian menswear brand.

STRICT RULES — follow every rule without exception:
- ONLY recommend products explicitly listed in the context
- NEVER invent, assume or mention products not in the context
- NEVER mention prices not in the context
- STRICTLY match the user's requested category — if user asks for trousers, ONLY recommend trousers
- STRICTLY respect the user's budget — NEVER recommend products above the stated price
- If no products match the exact category and budget, say so honestly
- Always use the exact product name as listed in the context
- Be concise, warm and helpful"""

    user_prompt = f"""Customer query: {user_query}

Available products:
{context}

Important: Only recommend products that exactly match the category and price the customer asked for."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Step 5 — Only return products actually mentioned in the answer
    matched_products = filter_products_by_mention(answer, retrieved_products)

    return {
        "query": user_query,
        "answer": answer,
        "products_used": matched_products
    }