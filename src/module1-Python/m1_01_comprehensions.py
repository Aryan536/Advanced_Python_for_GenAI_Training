# Real-world example: cleaning raw product titles scraped from an e-commerce site
raw_titles = ["  Wireless EARBUDS ", "smart watch", "", "  Yoga Mat", None, "AIR fryer  "]

# 1) List comprehension with a filter: keep valid titles and clean them
clean_titles = [t.strip().title() for t in raw_titles if t and t.strip()]
print("Clean titles:", clean_titles)

# 2) Dict comprehension: product -> price after a 10% festival discount
prices = {"Earbuds": 1999, "Smart Watch": 3999, "Yoga Mat": 899}
sale_prices = {name: round(price * 0.9, 2) for name, price in prices.items()}
print("Sale prices:", sale_prices)

# 3) Set comprehension: unique categories (duplicates removed automatically)
orders = [("Earbuds", "Electronics"), ("Yoga Mat", "Sports"), ("Watch", "Electronics")]
categories = {category for _, category in orders}
print("Unique categories:", sorted(categories))

# 4) Conditional expression inside a comprehension: label each order
order_values = [450, 2300, 999, 5600]
labels = ["High" if v > 2000 else "Low" for v in order_values]
print("Order labels:", labels)

# 5) Nested comprehension: flatten tokenised sentences (very common in NLP)
sentences = [["deliver", "fast"], ["great", "quality"], ["return", "item"]]
tokens = [word for sentence in sentences for word in sentence]
print("Flattened tokens:", tokens)
