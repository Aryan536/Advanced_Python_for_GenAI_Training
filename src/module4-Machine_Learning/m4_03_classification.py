import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# Real dataset: 569 tumour samples, 30 measurements each. Target: 0 = malignant, 1 = benign
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({"model": name,
                 "accuracy": accuracy_score(y_test, pred),
                 "precision": precision_score(y_test, pred),
                 "recall": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred),
                 "roc_auc": roc_auc_score(y_test, proba)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

best = models["Logistic Regression"]
pred = best.predict(X_test)
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=data.target_names))

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=data.target_names, cmap="Blues")
plt.title("Logistic Regression - confusion matrix")
plt.show()
