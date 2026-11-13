import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1) Eigenvalues and eigenvectors: A v = lambda v
A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eigh(A)    # eigh = for symmetric matrices
print("Eigenvalues:", eigenvalues)
v = eigenvectors[:, 1]                            # eigenvector for the largest eigenvalue
print("A @ v      :", A @ v)
print("lambda * v :", eigenvalues[1] * v, "(same direction, only scaled)")

# 2) SVD for recommendations: users x products rating matrix (0 = not rated)
ratings = np.array([[5, 4, 0, 1, 0],
                    [4, 5, 1, 0, 1],
                    [1, 0, 5, 4, 5],
                    [0, 1, 4, 5, 4],
                    [5, 5, 0, 0, 1]], dtype=float)
U, S, Vt = np.linalg.svd(ratings)
print("\nSingular values:", S.round(2))
k = 2                                             # keep the 2 strongest "taste" patterns
approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print("Rank-2 reconstruction (predicted ratings):\n", approx.round(1))
print("Predicted score of user 0 for product 2:", approx[0, 2].round(2), "(low -> do not recommend)")

# 3) PCA on a real dataset: 13 chemical features of wines -> 2 dimensions
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
print("\nExplained variance by PC1, PC2:", pca.explained_variance_ratio_.round(3),
      "| total:", pca.explained_variance_ratio_.sum().round(3))

for label in np.unique(wine.target):
    plt.scatter(X2[wine.target == label, 0], X2[wine.target == label, 1],
                label=wine.target_names[label], alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
plt.title("PCA: 13 features compressed to 2")
plt.show()
