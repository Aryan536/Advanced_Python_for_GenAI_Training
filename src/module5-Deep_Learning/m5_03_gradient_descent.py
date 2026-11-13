import numpy as np
import matplotlib.pyplot as plt

# Real-world example: learn  revenue = w * ad_spend + b  with gradient descent
rng = np.random.default_rng(0)
ad_spend = rng.uniform(0, 10, 100)                  # in Rs lakh
revenue = 3.0 * ad_spend + 5 + rng.normal(0, 2, 100)

def train(lr, epochs=1000):
    w, b = 0.0, 0.0
    losses = []
    n = len(ad_spend)
    for _ in range(epochs):
        pred = w * ad_spend + b
        error = pred - revenue
        loss = (error ** 2).mean()                  # Mean Squared Error
        dw = 2 / n * (error * ad_spend).sum()       # dLoss/dw
        db = 2 / n * error.sum()                    # dLoss/db
        w -= lr * dw                                # step downhill
        b -= lr * db
        losses.append(loss)
        if loss > 1e8:                              # stop if training blows up
            break
    return w, b, losses

for lr in [0.001, 0.005, 0.02, 0.04]:
    w, b, losses = train(lr)
    if losses[-1] > 1e8:
        print(f"lr={lr:<6} -> DIVERGED: learning rate too high, the loss exploded")
    else:
        print(f"lr={lr:<6} -> w={w:.2f}, b={b:.2f}, final loss={losses[-1]:.2f}")
    plt.plot(losses, label=f"lr={lr}")

print("True values: w=3.00, b=5.00")
plt.yscale("log"); plt.ylim(1, 1e4); plt.xlabel("Epoch"); plt.ylabel("MSE loss (log scale)")
plt.title("Gradient descent: effect of learning rate"); plt.legend()
plt.show()
