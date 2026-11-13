import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# 1) Embedding layer: maps each token id to a learnable dense vector
embedding = layers.Embedding(input_dim=1000, output_dim=8)     # vocab 1000, 8-D vectors
token_ids = np.array([[12, 45, 7, 3]])                          # one sentence of 4 tokens
vectors = embedding(token_ids).numpy()
print("Token ids shape:", token_ids.shape, "-> embeddings shape:", vectors.shape)

# 2) Scaled dot-product attention - the core operation inside every Transformer/LLM
#    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

tokens = ["the", "phone", "battery", "died"]
rng = np.random.default_rng(1)
d_model = 6
X = rng.normal(0, 1, (4, d_model))                 # pretend these are the 4 token embeddings
X[2] = X[1] * 0.9 + rng.normal(0, 0.2, d_model)    # make "battery" related to "phone"
X[3] = X[2] * 0.8 + rng.normal(0, 0.3, d_model)    # and "died" related to "battery"

Wq, Wk, Wv = (rng.normal(0, 0.5, (d_model, d_model)) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv                   # queries, keys, values

scores = Q @ K.T / np.sqrt(d_model)                # how much each token "looks at" others
weights = softmax(scores)                          # each row sums to 1
output = weights @ V                               # context-aware token representations

print("\nAttention weights (rows = query token):")
print("         " + "".join(f"{t:>9}" for t in tokens))
for t, row in zip(tokens, weights):
    print(f"{t:>8} " + "".join(f"{w:9.2f}" for w in row))
print("Output shape:", output.shape, "(same as input: 4 tokens x 6 dims)")

plt.imshow(weights, cmap="Blues")
plt.xticks(range(4), tokens); plt.yticks(range(4), tokens)
plt.colorbar(); plt.title("Self-attention weights")
plt.show()
