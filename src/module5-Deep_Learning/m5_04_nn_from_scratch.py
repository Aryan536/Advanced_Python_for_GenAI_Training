import numpy as np

# XOR cannot be solved by one neuron (not linearly separable) but a tiny
# 2-layer network can. This is forward pass + backpropagation in pure NumPy.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

rng = np.random.default_rng(42)
W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros((1, 4))     # input(2) -> hidden(4)
W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros((1, 1))     # hidden(4) -> output(1)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
lr = 0.5

for epoch in range(10001):
    # Forward pass
    h = np.tanh(X @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    loss = -np.mean(y * np.log(out + 1e-9) + (1 - y) * np.log(1 - out + 1e-9))

    # Backward pass (chain rule)
    d_out = (out - y) / len(X)                 # gradient of BCE + sigmoid
    dW2 = h.T @ d_out
    db2 = d_out.sum(axis=0, keepdims=True)
    d_h = (d_out @ W2.T) * (1 - h ** 2)        # derivative of tanh
    dW1 = X.T @ d_h
    db1 = d_h.sum(axis=0, keepdims=True)

    # Gradient descent update
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 2000 == 0:
        print(f"epoch {epoch:5d}  loss {loss:.4f}")

print("\nPredictions after training:")
for inp, p in zip(X, out.ravel()):
    print(f"  {inp} -> {p:.3f} -> class {int(p > 0.5)}")
