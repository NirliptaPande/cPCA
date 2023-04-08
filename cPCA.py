import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def cPCA(X, Y, alpha, n_components):
    """
    Perform contrastive PCA on the given target and background datasets.

    Parameters:
        X: Target dataset of shape (n_samples_X, n_features)
        Y: Background dataset of shape (n_samples_Y, n_features)
        alpha: Contrast parameter
        n_components: Number of components to keep

    Returns:
        V: Subspace spanned by the top k eigenvectors of C
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Calculate empirical covariance matrices
    Cx = np.cov(X_centered.T)
    Cy = np.cov(Y_centered.T)

    # Perform eigenvalue decomposition on C = (Cx - alpha*Cy)
    C = Cx - alpha * Cy
    w, v = np.linalg.eig(C)

    # Compute subspace spanned by top k eigenvectors of C
    idx = w.argsort()[::-1]
    V = v[:, idx[:n_components]]

    return V


def gen_data():
    n_features = 30
    n_subgroups = 4
    n_points_per_subgroup = 100
    n_background_points = 400

    target_data = np.zeros((n_subgroups * n_points_per_subgroup, n_features))
    background_data = np.zeros((n_background_points, n_features))

    for i in range(n_subgroups):
        start_index = i * n_points_per_subgroup
        end_index = start_index + n_points_per_subgroup

        # First 10 features
        target_data[start_index:end_index, :10] = np.random.normal(
            0, 10, (n_points_per_subgroup, 10))
        background_data[:, :10] = np.random.normal(
            0, 10, (n_background_points, 10))

        # Second 10 features
        if i == 1 or i == 2:  # green/blue
            target_data[start_index:end_index, 10:20] = np.random.normal(
                3, 1, (n_points_per_subgroup, 10))
        else:  # red/black
            target_data[start_index:end_index,
                        10:20] = np.random.normal(-1.5, 1, (n_points_per_subgroup, 10))
        background_data[:, 10:20] = np.random.normal(
            0, 3, (n_background_points, 10))

        # Last 10 features
        if i == 2 or i == 3:  # green/black
            target_data[start_index:end_index, -
                        10:] = np.random.normal(-1.5, 1, (n_points_per_subgroup, 10))
        else:  # red/blue
            target_data[start_index:end_index, -
                        10:] = np.random.normal(1.5, 1, (n_points_per_subgroup, 10))
        background_data[:, -
                        10:] = np.random.normal(0, 1, (n_background_points, 10))

    # Assign labels to subgroups
    labels = ['red', 'blue', 'green', 'black']
    subgroups = np.repeat(labels, n_points_per_subgroup)
    return target_data, background_data, subgroups


X, Y, subgroup_labels = gen_data()
# X -> (400,30)
# Y -> (400,30)
alpha = 2.15
n_components = 10
V_cPCA = cPCA(X, Y, alpha, n_components)
X_cPCA = X.dot(V_cPCA)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)


# Visualize projected data in two dimensions
colors = {'red': 'r', 'blue': 'b', 'green': 'g', 'black': 'k'}
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for subgroup in np.unique(subgroup_labels):
    i = np.where(subgroup_labels == subgroup)
    ax[0].scatter(X_cPCA[i, 0], X_cPCA[i, 1],
                  c=colors[subgroup], label=subgroup)
    ax[1].scatter(X_pca[i, 0], X_pca[i, 1], c=colors[subgroup], label=subgroup)

ax[0].set_title('cPCA')
ax[1].set_title('PCA')

plt.show()
