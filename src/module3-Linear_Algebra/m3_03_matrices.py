import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print("A @ B =\n", A @ B)                 # [[4 4] [10 8]]  (example from the primer)
print("B @ A =\n", B @ A, "\n(order matters: AB != BA)")
print("A transpose =\n", A.T)
print("det(A) =", round(np.linalg.det(A), 2))
A_inv = np.linalg.inv(A)
print("A inverse =\n", A_inv)
print("A @ A_inv =\n", np.round(A @ A_inv, 10), "(identity matrix)")

# Real-world example: find unit prices from three customers' bills (Ax = b)
# Customer 1 bought 2 tees, 1 jeans, 1 cap  -> Rs 3,096
# Customer 2 bought 1 tee, 2 jeans, 0 caps  -> Rs 3,497
# Customer 3 bought 1 tee, 0 jeans, 3 caps  -> Rs 2,296
quantities = np.array([[2, 1, 1],
                       [1, 2, 0],
                       [1, 0, 3]])
bills = np.array([3096, 3497, 2296])
unit_prices = np.linalg.solve(quantities, bills)
print("\nUnit prices [tee, jeans, cap]:", unit_prices.round(2))

# Matrix multiplication for many baskets at once
baskets = np.array([[1, 1, 0], [0, 1, 1], [4, 0, 0]])   # 3 new baskets
print("Basket totals:", baskets @ unit_prices)

# Broadcasting: apply a different discount to each product column in one step
discount = np.array([0.10, 0.20, 0.00])
print("Discounted prices per basket:\n", baskets * unit_prices * (1 - discount))
