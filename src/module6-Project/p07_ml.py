# STEP 7: Train and compare machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced_subsample",
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        class_weight="balanced", random_state=42),
}

def score_model(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {"model": name,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, proba)}

results, probas = [], {}
for name, model in models.items():
    model.fit(X_train_p, y_train)
    probas[name] = model.predict_proba(X_test_p)[:, 1]
    results.append(score_model(name, y_test, probas[name]))

results_df = pd.DataFrame(results).set_index("model").round(3)
print(results_df)

best_ml = results_df["roc_auc"].idxmax()
print(f"\nBest ML model by ROC-AUC: {best_ml}")
best_pred = (probas[best_ml] >= 0.5).astype(int)
print(classification_report(y_test, best_pred, target_names=["Kept", "Returned"]))
