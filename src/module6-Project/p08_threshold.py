# STEP 8: Choose a decision threshold that suits the business
# Calling a customer costs ~Rs 30; a prevented return saves ~Rs 250 of reverse logistics.
from sklearn.metrics import confusion_matrix

proba = probas[best_ml]
rows = []
for th in np.arange(0.30, 0.81, 0.05):
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= th).astype(int)).ravel()
    saving = tp * 250 * 0.4 - (tp + fp) * 30      # assume 40% of contacted risky orders are saved
    rows.append({"threshold": round(th, 2), "flagged": tp + fp, "caught_returns": tp,
                 "missed_returns": fn, "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn),
                 "net_saving_Rs": saving})
th_df = pd.DataFrame(rows).round(3)
print(th_df.to_string(index=False))
best_th = th_df.loc[th_df["net_saving_Rs"].idxmax(), "threshold"]
print(f"\nBest threshold for net saving: {best_th}")
