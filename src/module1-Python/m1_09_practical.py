import json
import re
from dataclasses import dataclass, asdict
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# PRACTICAL: a mini review-processing pipeline (the kind of pre-processing done
# before reviews are summarised by an LLM)
raw_reviews = [
    ("ORD1", "Electronics", "  GREAT earbuds!!! Sound quality is excellent  "),
    ("ORD2", "Fashion", "Size did not fit. Very disappointed, returning it."),
    ("ORD3", "Fashion", "Nice fabric and fast delivery :)"),
    ("ORD4", "Electronics", "Stopped working after 2 days. Poor quality."),
    ("ORD5", "Books", ""),
    ("ORD6", "Grocery", "Fresh and well packed, good value."),
    ("ORD7", "Electronics", "Okay product, average battery life."),
]

POSITIVE = {"great", "excellent", "nice", "fast", "fresh", "good", "love"}
NEGATIVE = {"poor", "disappointed", "returning", "stopped", "bad", "damaged", "late"}

@dataclass
class Review:
    order_id: str
    category: str
    text: str
    sentiment: str = "neutral"
    score: int = 0

def clean(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9\s]", "", text)        # remove punctuation and emojis

def score_sentiment(text):
    words = set(text.split())
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return label, score

def review_stream(rows):
    """Generator: yields cleaned Review objects and skips empty reviews."""
    for order_id, category, text in rows:
        cleaned = clean(text)
        if not cleaned:
            continue
        label, score = score_sentiment(cleaned)
        yield Review(order_id, category, cleaned, label, score)

reviews = list(review_stream(raw_reviews))
df = pd.DataFrame([asdict(r) for r in reviews])
print(df.to_string())

print("\nSentiment by category:")
print(pd.crosstab(df["category"], df["sentiment"]))

# Export negative reviews as JSON - ready to send to an LLM for root-cause summary
negatives = df[df["sentiment"] == "negative"].to_dict(orient="records")
with open("negative_reviews.json", "w") as f:
    json.dump(negatives, f, indent=2)
print(f"\nSaved {len(negatives)} negative reviews to negative_reviews.json")
