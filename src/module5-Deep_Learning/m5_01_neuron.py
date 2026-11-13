import numpy as np

# A single artificial neuron: z = w . x + b, output = activation(z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Real-world example: will this customer click "Buy"?
# features x = [items in cart, minutes on page, has coupon (0/1)]
x = np.array([3, 7.5, 1])
w = np.array([0.4, 0.15, 1.2])     # learned importance of each feature
b = -2.5                            # learned bias (baseline tendency)

z = np.dot(w, x) + b
p = sigmoid(z)
print(f"z = w.x + b = {z:.3f}")
print(f"P(buy) = sigmoid(z) = {p:.3f} -> {'BUY' if p >= 0.5 else 'NO BUY'}")

# The same neuron applied to many customers at once (matrix form)
X = np.array([[3, 7.5, 1],
              [0, 1.0, 0],
              [1, 3.0, 1],
              [5, 12.0, 0]])
probs = sigmoid(X @ w + b)
for row, pr in zip(X, probs):
    print(f"cart={row[0]:.0f}, minutes={row[1]:>4}, coupon={row[2]:.0f} -> P(buy)={pr:.2f}")
