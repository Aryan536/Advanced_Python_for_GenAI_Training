# STEP 14 (GenAI bridge): semantic search over customer reviews
# This is the "retrieval" half of a RAG chatbot that answers: "Why are customers returning X?"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cols = ["Order_ID", "Category", "Review_Text", "Returned"]
reviews = df.loc[df["Review_Text"] != "No review", cols].reset_index(drop=True)
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
review_vectors = vectorizer.fit_transform(reviews["Review_Text"])
print("Review vectors:", review_vectors.shape)

def find_reviews(query, top_k=5, category=None):
    scores = cosine_similarity(vectorizer.transform([query]), review_vectors).ravel()
    result = reviews.assign(score=scores)
    if category:
        result = result[result["Category"] == category]
    result = result.sort_values("score", ascending=False).drop_duplicates("Review_Text")
    return result.head(top_k)

print("\nQuery: 'size did not fit' in Fashion")
show = ["Order_ID", "Review_Text", "score"]
print(find_reviews("size did not fit", category="Fashion")[show].to_string(index=False))

print("\nQuery: 'damaged product not working' in Electronics")
print(find_reviews("damaged product not working", category="Electronics")[show].to_string(index=False))

# Build an LLM-ready prompt from returned-order reviews (send it to any LLM API)
sample = (reviews[(reviews["Category"] == "Fashion") & (reviews["Returned"] == "Yes")]["Review_Text"]
          .drop_duplicates().sample(10, random_state=1))
context = "\n".join("- " + t for t in sample)
prompt = ("You are an e-commerce analyst. Using only the reviews below, list the top reasons "
          f"customers return Fashion products and suggest 3 fixes.\n\nReviews:\n{context}")
print("\nPrompt for the LLM:\n", prompt)
