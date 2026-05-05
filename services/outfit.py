from groq import Groq
from services.embedder import generate_embedding
from services.vector_store import search_from_state
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

# Category groupings — update these if you add new categories later
TOP_CATEGORIES = ["shirts", "t-shirts"]
BOTTOM_CATEGORIES = ["jeans", "trousers"]


def filter_by_category_and_budget(
    products: list[dict],
    category_group: list[str],
    budget: float
) -> list[dict]:
    """
    Filters products by category group and budget.
    Uses sale_price if available, otherwise base_price.
    """
    filtered = []
    for p in products:
        category = p.get('category_name', '').lower()
        price = float(p.get('sale_price') or p.get('base_price') or 0)

        is_in_category = any(cat in category for cat in category_group)
        is_within_budget = price <= budget

        if is_in_category and is_within_budget:
            filtered.append(p)

    return filtered


def generate_outfit(
    occasion: str,
    budget: float,
    index,
    products: list[dict],
    embeddings
) -> dict:
    """
    Full outfit generation pipeline:
    1. Embed the occasion
    2. Search tops and bottoms separately within budget
    3. Pass to Groq to select best combination and write styling advice
    """

    # Step 1 — Embed the occasion query
    query_text = f"formal outfit for {occasion}"
    query_vector = generate_embedding(query_text)

    # Step 2 — Get all similar products from FAISS
    all_results = search_from_state(index, products, query_vector, top_k=19)

    # Step 3 — Filter tops and bottoms within budget
    # Budget is split — 60% for top, 40% for bottom (typical menswear ratio)
    top_budget = budget * 0.6
    bottom_budget = budget * 0.4

    tops = filter_by_category_and_budget(all_results, TOP_CATEGORIES, top_budget)
    bottoms = filter_by_category_and_budget(all_results, BOTTOM_CATEGORIES, bottom_budget)

    # Step 4 — Handle cases where budget split is too strict
    # If nothing found within split budget, try full budget for each
    if not tops:
        tops = filter_by_category_and_budget(all_results, TOP_CATEGORIES, budget)
    if not bottoms:
        bottoms = filter_by_category_and_budget(all_results, BOTTOM_CATEGORIES, budget)

    # Step 5 — Build context for Groq
    def format_items(items):
        lines = []
        for p in items:
            price = float(p.get('sale_price') or p.get('base_price'))
            lines.append(
                f"- {p['title']} | Brand: {p['brand']} | "
                f"Price: Rs.{int(price)} | Colors: {p.get('colors', 'N/A')} | "
                f"Sizes: {p.get('sizes', 'N/A')}"
            )
        return "\n".join(lines) if lines else "No products available in this category within budget."

    tops_context = format_items(tops)
    bottoms_context = format_items(bottoms)

    # Step 6 — Call Groq
    system_prompt = """You are a professional men's fashion stylist for Trendio, 
a premium Indian menswear brand.

STRICT RULES:
- ONLY select products from the provided lists
- NEVER invent products or prices
- Select exactly ONE top and ONE bottom
- Combined price must not exceed the total budget
- Give practical, confident styling advice
- Be concise and specific"""

    user_prompt = f"""Create a complete formal outfit for: {occasion}
Total budget: Rs.{int(budget)}

Available tops (shirts/t-shirts):
{tops_context}

Available bottoms (jeans/trousers):
{bottoms_context}

Respond in this exact format:
TOP: [product name] - Rs.[price]
BOTTOM: [product name] - Rs.[price]
TOTAL: Rs.[combined price]
STYLING ADVICE: [2-3 sentences of practical styling tips]"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    advice = response.choices[0].message.content

    return {
        "occasion": occasion,
        "budget": budget,
        "available_tops": tops,
        "available_bottoms": bottoms,
        "outfit_suggestion": advice
    }