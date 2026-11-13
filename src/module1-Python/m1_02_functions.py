# Real-world example: a reusable prompt builder for an LLM application
def build_prompt(question, context="", role="helpful shopping assistant", **options):
    """Return a prompt string. Extra keyword options become instructions."""
    prompt = f"You are a {role}.\n"
    if context:
        prompt += f"Use only this context:\n{context}\n"
    prompt += f"Question: {question}\n"
    for key, value in options.items():
        prompt += f"Instruction - {key}: {value}\n"
    return prompt

print(build_prompt("What is the return window?",
                   context="Products can be returned within 7 days of delivery.",
                   tone="friendly", max_words=40))

# *args collects any number of positional values into a tuple
def total_bill(*item_prices, gst_rate=0.18):
    subtotal = sum(item_prices)
    return round(subtotal * (1 + gst_rate), 2)

print("Bill with GST:", total_bill(499, 1299, 250))

# lambda + sorted / filter / map on a product list
products = [
    {"name": "Earbuds", "price": 1999, "rating": 4.3},
    {"name": "Yoga Mat", "price": 899, "rating": 4.6},
    {"name": "Air Fryer", "price": 5999, "rating": 4.1},
    {"name": "Green Tea", "price": 349, "rating": 4.4},
]
top_rated = sorted(products, key=lambda p: p["rating"], reverse=True)
under_1000 = list(filter(lambda p: p["price"] < 1000, products))
names_upper = list(map(lambda p: p["name"].upper(), products))

print("Top rated:", [p["name"] for p in top_rated])
print("Under Rs 1000:", [p["name"] for p in under_1000])
print("Upper-case names:", names_upper)
