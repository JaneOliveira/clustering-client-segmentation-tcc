"""
plotsconfig.py
---------------------------------
Módulo de funções de visualização para resultados de clusterização.
Suporta gráficos 2D e 3D genéricos, com PCA opcional e centróides.
---------------------------------
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# -------------------------------------------------------------------------
# Função auxiliar: cria colormap dinâmico
# -------------------------------------------------------------------------
def get_dynamic_colormap(n_clusters):
    """Retorna colormap ajustado para número de clusters."""
    if n_clusters <= 10:
        return plt.get_cmap("tab10", n_clusters)
    elif n_clusters <= 20:
        return plt.get_cmap("tab20", n_clusters)
    else:
        # gera gradiente contínuo de cores
        return ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_clusters)))

# -------------------------------------------------------------------------
# Plot 2D
# -------------------------------------------------------------------------
def plot_clusters_2d(X, labels, title="Cluster Plot (2D)", pca=False, x=0, y=1, centroids=None):
    """Plota clusters em 2D com PCA opcional e centróides."""
    X = np.asarray(X)
    labels = np.asarray(labels)

    if pca and X.shape[1] > 2:
        X = PCA(n_components=2).fit_transform(X)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    cmap = get_dynamic_colormap(n_clusters)

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        mask = labels == label
        color = 'gray' if label == -1 else cmap(label % max(n_clusters, 1))
        plt.scatter(X[mask, x], X[mask, y], c=[color], s=40, edgecolor='k', alpha=0.8, label=f"Cluster {label}" if label != -1 else "Ruído (-1)")

    # Centróides (opcional)
    if centroids is not None:
        centroids = np.asarray(centroids)
        if pca and centroids.shape[1] > 2:
            centroids = PCA(n_components=2).fit_transform(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolor='k', label='Centroides')

    plt.title(title)
    plt.xlabel("Componente 1" if pca else "Feature 1")
    plt.ylabel("Componente 2" if pca else "Feature 2")
    plt.legend(title="Clusters", loc="best")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Plot 3D
# -------------------------------------------------------------------------
def plot_clusters_3d(X, labels, title="Clusters 3D", pca=False, x=0, y=1, z=2, centroids=None):
    """Plota clusters em 3D com PCA opcional, centróides e paleta dinâmica."""
    X = np.asarray(X)
    if pca and X.shape[1] > 3:
        X = PCA(n_components=3).fit_transform(X)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    labels = np.array(labels)[mask]

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    cmap = get_dynamic_colormap(max(n_clusters, 1))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for label in unique_labels:
        mask = labels == label
        color = 'gray' if label == -1 else cmap(label % max(n_clusters, 1))
        ax.scatter(X[mask, x], X[mask, y], X[mask, z], c=[color], s=40, edgecolor='k', alpha=0.85, label=f"Cluster {label}" if label != -1 else "Ruído (-1)")

    # Centróides (opcional)
    if centroids is not None:
        centroids = np.asarray(centroids)
        if pca and centroids.shape[1] > 3:
            centroids = PCA(n_components=3).fit_transform(centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   c='red', marker='X', s=250, edgecolor='k', label='Centroides')

    ax.set_title(title)
    ax.set_xlabel("Componente 1" if pca else "Feature 1")
    ax.set_ylabel("Componente 2" if pca else "Feature 2")
    ax.set_zlabel("Componente 3" if pca else "Feature 3")

    ax.legend(title="Clusters", loc="best")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.show()
