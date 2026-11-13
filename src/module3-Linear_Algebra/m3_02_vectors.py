import numpy as np

u = np.array([3, 4])
v = np.array([6, 0])

print("u + v =", u + v)                 # [9 4]
print("u - v =", u - v)                 # [-3 4]
print("2u    =", 2 * u)                 # [6 8]

# Norms (vector length)
x = np.array([3, -4, 12])
print("\nL1 norm  :", np.linalg.norm(x, 1))        # |3|+|-4|+|12| = 19
print("L2 norm  :", np.linalg.norm(x))           # sqrt(9+16+144) = 13
print("Inf norm :", np.linalg.norm(x, np.inf))   # max |x_i| = 12

# Dot product (example from the primer)
x = np.array([2, -1, 5])
w = np.array([1, 3, -2])
print("\nx . w =", np.dot(x, w), "| same as x @ w =", x @ w)    # -11

# Cosine similarity (example from the primer): u=[3,4], v=[6,0] -> 0.6
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("cos(u, v) =", round(cosine_similarity(u, v), 2))

# Real-world example: toy 4-D "embeddings" of products
# dims ~ [electronics-ness, fashion-ness, sports-ness, price level]
emb = {
    "wireless earbuds": np.array([0.90, 0.05, 0.20, 0.40]),
    "bluetooth speaker": np.array([0.85, 0.02, 0.10, 0.45]),
    "running shoes":     np.array([0.05, 0.60, 0.80, 0.50]),
    "yoga mat":          np.array([0.02, 0.10, 0.90, 0.20]),
    "denim jeans":       np.array([0.01, 0.95, 0.05, 0.40]),
}
query = "wireless earbuds"
scores = {name: cosine_similarity(emb[query], vec) for name, vec in emb.items() if name != query}
print(f"\nProducts most similar to '{query}':")
for name, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"  {name:18} {s:.3f}")
