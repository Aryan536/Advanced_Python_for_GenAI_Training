import sys

# Real-world example: splitting a long document into overlapping chunks before
# sending it to an embedding model (this is step 1 of every RAG pipeline)
def chunk_text(text, chunk_size=10, overlap=3):
    """Yield chunks of `chunk_size` words that overlap by `overlap` words."""
    words = text.split()
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        yield " ".join(words[start:start + chunk_size])
        if start + chunk_size >= len(words):
            break

policy = ("Customers can return most products within 7 days of delivery. "
          "Electronics must be unused and in the original packaging. "
          "Refunds are processed to the original payment method within 5 business days. "
          "Cash on delivery orders are refunded to the bank account given by the customer.")

for i, chunk in enumerate(chunk_text(policy), start=1):
    print(f"Chunk {i}: {chunk}")

# Batching: send reviews to an API in groups instead of one huge request
def batch(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

reviews = [f"review_{i}" for i in range(1, 11)]
for b in batch(reviews, 4):
    print("Sending batch:", b)

# Why generators? They produce values lazily and use almost no memory
squares_list = [x * x for x in range(1_000_000)]
squares_gen = (x * x for x in range(1_000_000))
print(f"List size: {sys.getsizeof(squares_list):,} bytes")
print(f"Generator size: {sys.getsizeof(squares_gen):,} bytes")
