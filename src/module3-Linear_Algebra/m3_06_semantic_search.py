import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PRACTICAL: mini product search engine = the "retrieval" step of RAG
catalog = pd.DataFrame({
    "product": ["Wireless Earbuds", "Bluetooth Speaker", "Running Shoes", "Yoga Mat",
                "Denim Jeans", "Air Fryer", "Mixer Grinder", "Face Serum", "Sunscreen SPF50",
                "Smart Watch"],
    "description": [
        "bluetooth wireless earbuds with noise cancellation and long battery life",
        "portable bluetooth speaker with deep bass and waterproof body",
        "lightweight running shoes with cushioned sole for jogging and gym",
        "non slip yoga mat for home workout, gym and stretching",
        "slim fit blue denim jeans for men, stretchable cotton fabric",
        "air fryer for oil free cooking, crispy fries and snacks",
        "powerful mixer grinder with three jars for chutney and juice",
        "vitamin c face serum for glowing skin and dark spots",
        "sunscreen spf 50 for skin protection from sun, non greasy",
        "smart watch with heart rate, step tracking and bluetooth calling",
    ],
})

# Step 1: turn every description into a vector (a matrix: products x vocabulary)
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(catalog["description"])
print("Document-term matrix shape:", doc_vectors.shape)

# Step 2: turn the user's query into a vector and rank products by cosine similarity
def search(query, top_k=3):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, doc_vectors).ravel()
    best = [i for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]
    return catalog.iloc[best].assign(score=scores[best].round(3))[["product", "score"]]

for q in ["something for my gym workout", "bluetooth music", "skin care for sun"]:
    print(f"\nQuery: '{q}'")
    print(search(q).to_string(index=False))

# Step 3: verify the library with our own NumPy cosine formula
q = vectorizer.transform(["bluetooth music"]).toarray()[0]
d = doc_vectors[1].toarray()[0]
manual = q @ d / (np.linalg.norm(q) * np.linalg.norm(d))
print(f"\nManual cosine for 'Bluetooth Speaker' = {manual:.3f}")
