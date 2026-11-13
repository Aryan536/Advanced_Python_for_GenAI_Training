import numpy as np

# Real-world example: a fraud model flags a payment. How likely is it really fraud?
p_fraud = 0.01                 # prior: 1% of payments are fraudulent
p_flag_given_fraud = 0.95      # sensitivity: model catches 95% of fraud
p_flag_given_genuine = 0.05    # false-alarm rate on genuine payments

# Total probability of a flag
p_flag = p_flag_given_fraud * p_fraud + p_flag_given_genuine * (1 - p_fraud)

# Bayes' theorem: P(fraud | flag) = P(flag | fraud) * P(fraud) / P(flag)
p_fraud_given_flag = p_flag_given_fraud * p_fraud / p_flag
print(f"P(flag)          = {p_flag:.4f}")
print(f"P(fraud | flag)  = {p_fraud_given_flag:.3f}")
print(f"-> only ~{p_fraud_given_flag:.0%} of flagged payments are real fraud")

# Verify with a simulation of 1,000,000 payments
rng = np.random.default_rng(0)
n = 1_000_000
is_fraud = rng.random(n) < p_fraud
flagged = np.where(is_fraud, rng.random(n) < p_flag_given_fraud, rng.random(n) < p_flag_given_genuine)
print(f"Simulated P(fraud | flag) = {is_fraud[flagged].mean():.3f}")

# Conditional probability from a table: P(return | COD) vs P(return | prepaid)
cod_orders, cod_returns = 2000, 520
prepaid_orders, prepaid_returns = 8000, 1200
print(f"\nP(return | COD)     = {cod_returns / cod_orders:.2f}")
print(f"P(return | prepaid) = {prepaid_returns / prepaid_orders:.2f}")
