import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# 1) Overfitting: as the tree gets deeper, train accuracy -> 100% but test accuracy stalls
depths = range(1, 16)
train_acc, test_acc = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
    train_acc.append(tree.score(X_train, y_train))
    test_acc.append(tree.score(X_test, y_test))

plt.plot(depths, train_acc, "o-", label="Train accuracy")
plt.plot(depths, test_acc, "s-", label="Test accuracy")
plt.xlabel("Tree depth (model complexity)"); plt.ylabel("Accuracy")
plt.title("Underfitting (left) vs overfitting (right)"); plt.legend()
plt.show()

# 2) 5-fold cross-validation: a more reliable estimate than a single split
rf = RandomForestClassifier(n_estimators=200, random_state=0)
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print("CV accuracy per fold:", scores.round(3))
print(f"Mean = {scores.mean():.3f} +/- {scores.std():.3f}")

# 3) Hyper-parameter tuning with GridSearchCV
grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    param_grid={"max_depth": [2, 3, 4, 5, 6, None],
                                "min_samples_leaf": [1, 5, 10]},
                    cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.3f} | Test accuracy: {grid.score(X_test, y_test):.3f}")
