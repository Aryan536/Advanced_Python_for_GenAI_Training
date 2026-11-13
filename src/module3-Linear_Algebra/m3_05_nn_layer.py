import numpy as np

# One dense neural-network layer is just linear algebra: Z = XW + b, A = ReLU(Z)
rng = np.random.default_rng(0)
X = np.array([[0.5, 1.2, -0.3],          # 2 samples (batch), 3 input features each
              [1.0, -0.7, 2.0]])
W = rng.normal(0, 0.5, size=(3, 4))      # 3 inputs -> 4 neurons
b = np.zeros(4)                          # one bias per neuron

Z = X @ W + b                            # (2x3) @ (3x4) + (4,) -> (2x4), bias is broadcast
A = np.maximum(0, Z)                     # ReLU activation
print("X shape", X.shape, "| W shape", W.shape, "| output shape", A.shape)
print("Z (before activation):\n", Z.round(3))
print("A (after ReLU):\n", A.round(3))
