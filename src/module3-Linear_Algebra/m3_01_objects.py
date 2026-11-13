import numpy as np

scalar = np.array(5)                                   # 0-D: e.g. one loss value
vector = np.array([1, 2, 3])                           # 1-D: e.g. one customer's features
matrix = np.array([[1, 2, 3], [4, 5, 6]])              # 2-D: e.g. a table of customers x features
tensor = np.zeros((32, 3, 224, 224))                   # 4-D: a batch of 32 RGB images, 224x224

for name, obj in [("Scalar", scalar), ("Vector", vector), ("Matrix", matrix), ("Tensor", tensor)]:
    print(f"{name:7}: ndim={obj.ndim}, shape={obj.shape}")

# Real-world example: 4 customers described by [age, orders, avg_spend]
customers = np.array([[25, 3, 1200],
                      [41, 12, 3400],
                      [33, 7, 2100],
                      [19, 1, 600]])
print("\nCustomer matrix shape:", customers.shape, "-> 4 rows (samples) x 3 columns (features)")
print("Second customer vector:", customers[1])
print("All ages (column 0):   ", customers[:, 0])

# Reshape: the same data viewed with a different shape (common before feeding a model)
flat = customers.reshape(-1)          # -1 means "work it out for me"
print("Flattened:", flat, "shape", flat.shape)
print("Back to 2 x 6:\n", flat.reshape(2, 6))
