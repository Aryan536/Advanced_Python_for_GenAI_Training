import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 200)
activations = {
    "Sigmoid (0 to 1)": 1 / (1 + np.exp(-z)),
    "Tanh (-1 to 1)": np.tanh(z),
    "ReLU max(0, z)": np.maximum(0, z),
    "Leaky ReLU": np.where(z > 0, z, 0.1 * z),
}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (name, values) in zip(axes, activations.items()):
    ax.plot(z, values, linewidth=2)
    ax.axhline(0, color="grey", linewidth=0.5); ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# Softmax: turns raw scores (logits) into probabilities that sum to 1.
# This is exactly how an LLM picks the next word from its vocabulary.
def softmax(logits, temperature=1.0):
    scaled = np.array(logits) / temperature
    e = np.exp(scaled - scaled.max())          # subtract max for numerical stability
    return e / e.sum()

words = ["delivered", "late", "broken", "excellent"]
logits = [2.0, 1.0, 0.1, 1.5]
for t in [0.5, 1.0, 2.0]:
    probs = softmax(logits, temperature=t)
    print(f"temperature={t}: " + ", ".join(f"{w}={p:.2f}" for w, p in zip(words, probs)))
print("Low temperature = confident/deterministic, high temperature = more random/creative")
