# STEP 10: Compare all models with ROC curves and confusion matrices
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for name, pr in probas.items():
    fpr, tpr, _ = roc_curve(y_test, pr)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, pr):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", label="Random guess")
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves"); axes[0].legend()

ConfusionMatrixDisplay.from_predictions(y_test, (probas[best_ml] >= 0.5).astype(int),
                                        display_labels=["Kept", "Returned"], cmap="Blues", ax=axes[1])
axes[1].set_title(f"{best_ml} - confusion matrix")
plt.tight_layout()
plt.show()
print(results_df.sort_values("roc_auc", ascending=False))
