import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# PRACTICAL: recognise handwritten digits (think: reading PIN codes on shipping labels)
digits = load_digits()                       # 1,797 real images of 8x8 pixels
print("Data shape:", digits.data.shape, "-> each image is flattened to 64 numbers")

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for ax, img, label in zip(axes, digits.images, digits.target):
    ax.imshow(img, cmap="gray_r"); ax.set_title(label); ax.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))
print(pd.Series(results).sort_values(ascending=False).round(4))

best_name = max(results, key=results.get)
best = models[best_name]
pred = best.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
plt.title(f"{best_name} - confusion matrix")
plt.show()

wrong = np.where(pred != y_test)[0]
print(f"{best_name} misclassified {len(wrong)} of {len(y_test)} test images")
fig, axes = plt.subplots(1, min(6, len(wrong)), figsize=(10, 2), squeeze=False)
for ax, i in zip(axes[0], wrong[:6]):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"true {y_test[i]} / pred {pred[i]}", fontsize=8); ax.axis("off")
plt.show()
