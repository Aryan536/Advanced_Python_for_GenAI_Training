import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Binomial: 20 visitors, each buys with probability 0.10
binom = stats.binom(n=20, p=0.10)
print(f"Binomial  P(exactly 2 buy)   = {binom.pmf(2):.3f}")
print(f"Binomial  P(at least 3 buy)  = {1 - binom.cdf(2):.3f}")

# Geometric: how many visitors until the first purchase?
geom = stats.geom(p=0.10)
print(f"Geometric P(first sale within 5 visitors) = {geom.cdf(5):.3f}")

# Poisson: on average 4 orders arrive per minute
pois = stats.poisson(mu=4)
print(f"Poisson   P(more than 8 orders in a minute) = {1 - pois.cdf(8):.4f}")

# Normal: delivery time ~ N(mean=3 days, sd=1 day)
norm = stats.norm(loc=3, scale=1)
print(f"Normal    P(delivery > 5 days) = {1 - norm.cdf(5):.4f}")
print(f"Normal    95% of deliveries fall between {norm.ppf(0.025):.2f} and {norm.ppf(0.975):.2f} days")

# Exponential: time between orders, average 15 seconds
expo = stats.expon(scale=15)
print(f"Exponential P(next order within 5 s) = {expo.cdf(5):.3f}")

# Central Limit Theorem: averages of a skewed variable become normal
rng = np.random.default_rng(0)
population = rng.exponential(scale=15, size=100_000)          # very skewed
sample_means = [rng.choice(population, 50).mean() for _ in range(3000)]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
k = np.arange(0, 11)
axes[0, 0].bar(k, binom.pmf(k)); axes[0, 0].set_title("Binomial(n=20, p=0.1)")
axes[0, 1].bar(k + 1, geom.pmf(k + 1)); axes[0, 1].set_title("Geometric(p=0.1)")
axes[0, 2].bar(np.arange(0, 13), pois.pmf(np.arange(0, 13))); axes[0, 2].set_title("Poisson(lambda=4)")
x = np.linspace(-1, 7, 200)
axes[1, 0].plot(x, norm.pdf(x)); axes[1, 0].set_title("Normal(3, 1)")
axes[1, 1].hist(population, bins=60, color="orange"); axes[1, 1].set_title("Skewed population")
axes[1, 2].hist(sample_means, bins=40, color="green")
axes[1, 2].set_title("CLT: means of samples (n=50)")
plt.tight_layout()
plt.show()
