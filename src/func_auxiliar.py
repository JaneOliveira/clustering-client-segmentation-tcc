#Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os
import sys
import pandas as pd
from time import time

from sklearn.datasets import make_blobs
from scipy.spatial.distance import jensenshannon

notebook_dir = os.getcwd()
src_path = os.path.abspath(os.path.join(notebook_dir, '..', 'src'))
sys.path.append(src_path)
from dbscan import DBSCAN
from kmeans import KMeans
from kmedoids import KMedoids
from wardmethod import WardMethod
from expectationmaximization import ExpectationMaximization
import distance_metrics
from evaluationmetrics import EvaluationMetrics
from mpl_toolkits.mplot3d import Axes3D  # só para garantir o registro do proj. 3D


def generate_synthetic_blobs(
    n_samples=1000,
    centers=3,
    cluster_std=0.7,
    random_state=42,
):
    """
    Gera um dataset sintético simples (blobs) só para validar se
    os algoritmos estão funcionando como esperado.
    """
    X, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return X, labels

def calculate_clustering_metrics(algorithm_name, metric_name, runtime, X, labels, eval_metric='euclidean_distance', eval_kwargs=None):
    """
    Calcula as métricas de clusterização.
    Agora aceita 'eval_metric' e 'eval_kwargs' para calcular o Silhouette correto.
    """
    if eval_kwargs is None:
        eval_kwargs = {}

    silhouette_metric_name = {
    'euclidean_distance': 'euclidean',
    'manhattan_distance': 'manhattan',
    'cosine_distance': 'cosine',
    'mahalanobis_distance': 'mahalanobis',
    'jensen-shannon_distance': jensenshannon
}
    # Repassamos a métrica (eval_metric) e os argumentos extras (eval_kwargs) para a classe
    evaluator = EvaluationMetrics(X, labels, y_true=None, metric=silhouette_metric_name[eval_metric], **eval_kwargs)
    
    sse = evaluator.sse_euclidean()
    sil = evaluator.silhouette()      # Agora vai usar a métrica correta internamente
    dbi = evaluator.davies_bouldin()
    ch = evaluator.calinski_harabasz()

    return {
        "algorithm": algorithm_name,
        "internal_metric": metric_name,
        "n_clusters_found": len(set(labels)) - (1 if -1 in labels else 0),
        "runtime_sec": runtime,
        "sse_euclidean": sse,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
    }


def plot_algorithm_all_metrics(
    plot_dict,
    algorithm_name,
    x_index=0,
    y_index=1,
    n_cols=3,
    figsize=(16, 10),
    feature_names=None,
):
    """
    Gera uma figura ÚNICA com vários subplots,
    um para cada métrica de distância usada no algoritmo escolhido.

    plot_dict deve ter o formato:
    {
        "euclidiana": { "X": X, "labels": labels, "model": modelo },
        "manhattan":  { "X": X, "labels": labels, "model": modelo },
        ...
    }

    X pode ser numpy array OU DataFrame.
    """

    metrics = list(plot_dict.keys())
    n_metrics = len(metrics)

    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).ravel()

    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        data = plot_dict[metric_name]

        # Garante NumPy array mesmo se vier DataFrame
        X = np.asarray(data["X"])
        labels = np.asarray(data["labels"])
        model = data["model"]

        # Seleção das duas features para o plot
        X_plot = X[:, [x_index, y_index]]

        # Scatter dos pontos
        ax.scatter(
            X_plot[:, 0], X_plot[:, 1],
            c=labels,
            cmap="viridis",
            s=30,
            edgecolors="k",
            alpha=0.8
        )

        # ----- CENTROS -----
        if model is not None:
            # K-Means (.centroids)
            if hasattr(model, "centroids") and model.centroids is not None:
                centers = np.asarray(model.centroids)
                # assume que os centróides estão no mesmo espaço de X
                if centers.shape[1] > max(x_index, y_index):
                    centers_plot = centers[:, [x_index, y_index]]
                    ax.scatter(
                        centers_plot[:, 0], centers_plot[:, 1],
                        c="red", marker="X", s=120, label="Centroides"
                    )

            # K-Medoids (.medoids)
            elif hasattr(model, "medoids") and model.medoids is not None:
                centers = np.asarray(model.medoids)
                if centers.shape[1] > max(x_index, y_index):
                    centers_plot = centers[:, [x_index, y_index]]
                    ax.scatter(
                        centers_plot[:, 0], centers_plot[:, 1],
                        c="red", marker="P", s=120, label="Medoides"
                    )

            # EM (.means_)
            elif hasattr(model, "means_") and model.means_ is not None:
                centers = np.asarray(model.means_)
                # aqui trato igual: projeto nas mesmas features se tiver dimensão suficiente
                if centers.shape[1] > max(x_index, y_index):
                    centers_plot = centers[:, [x_index, y_index]]
                    ax.scatter(
                        centers_plot[:, 0], centers_plot[:, 1],
                        c="red", marker="X", s=120, label="Médias (μ_k)"
                    )

        ax.set_title(f"Métrica: {metric_name}")
        ax.set_xlabel(f"Feature {x_index} - {feature_names[x_index] if feature_names else ''}")
        ax.set_ylabel(f"Feature {y_index} - {feature_names[y_index] if feature_names else ''}")
        ax.grid(True)

        handles, labels_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    # Remove subplots sobrando, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"{algorithm_name} — Comparação entre métricas de distância",
        fontsize=16,
        y=1.02
    )

    plt.tight_layout()
    plt.show()


def plot_algorithm_all_metrics_3d(
    plot_dict,
    algorithm_name,
    x_index=0,
    y_index=1,
    z_index=2,
    n_cols=3,
    figsize=(18, 10),
    feature_names=None,
    elev=20,
    azim=45,
):
    """
    Versão 3D do plot_algorithm_all_metrics:
    Gera uma figura ÚNICA com vários subplots 3D,
    um para cada métrica de distância usada no algoritmo escolhido.

    plot_dict:
    {
        "euclidean_distance": { "X": X, "labels": labels, "model": modelo },
        "manhattan_distance": { "X": X, "labels": labels, "model": modelo },
        ...
    }

    X pode ser numpy array OU DataFrame.
    """

    metrics = list(plot_dict.keys())
    n_metrics = len(metrics)

    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        subplot_kw={"projection": "3d"}
    )

    axes = np.array(axes).ravel()

    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        data = plot_dict[metric_name]

        # Garante NumPy array mesmo se vier DataFrame
        X = np.asarray(data["X"])
        labels = np.asarray(data["labels"])
        model = data["model"]

        # Seleção das três features para o plot
        X_plot = X[:, [x_index, y_index, z_index]]

        # Scatter dos pontos com cor por rótulo de cluster
        scatter = ax.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            X_plot[:, 2],
            c=labels,
            cmap="viridis",
            s=20,
            edgecolors="none",
            alpha=0.8,
        )

        # --------- LEGENDA DOS CLUSTERS ---------
        unique_labels = np.unique(labels)

        for lab in unique_labels:
            # Nome amigável do cluster
            if lab == -1:
                # Tipicamente DBSCAN
                cluster_name = "Outliers"
            else:
                cluster_name = f"Cluster {lab}"

            # Cor correspondente na colormap
            color = scatter.cmap(scatter.norm(lab))

            # Ponto "fantasma" só para criar entrada na legenda
            ax.scatter(
                [], [], [],  # nada é desenhado, só serve pra legenda
                c=[color],
                marker="o",
                label=cluster_name,
            )

        # ----- CENTROS -----
        if model is not None:
            # K-Means (.centroids)
            if hasattr(model, "centroids") and model.centroids is not None:
                centers = np.asarray(model.centroids)
                if centers.shape[1] > max(x_index, y_index, z_index):
                    centers_plot = centers[:, [x_index, y_index, z_index]]
                    ax.scatter(
                        centers_plot[:, 0],
                        centers_plot[:, 1],
                        centers_plot[:, 2],
                        c="red",
                        marker="X",
                        s=80,
                        label="Centroides",
                    )

            # K-Medoids (.medoids)
            elif hasattr(model, "medoids") and model.medoids is not None:
                centers = np.asarray(model.medoids)
                if centers.shape[1] > max(x_index, y_index, z_index):
                    centers_plot = centers[:, [x_index, y_index, z_index]]
                    ax.scatter(
                        centers_plot[:, 0],
                        centers_plot[:, 1],
                        centers_plot[:, 2],
                        c="red",
                        marker="P",
                        s=80,
                        label="Medoides",
                    )

            # EM (.means_)
            elif hasattr(model, "means_") and model.means_ is not None:
                centers = np.asarray(model.means_)
                if centers.shape[1] > max(x_index, y_index, z_index):
                    centers_plot = centers[:, [x_index, y_index, z_index]]
                    ax.scatter(
                        centers_plot[:, 0],
                        centers_plot[:, 1],
                        centers_plot[:, 2],
                        c="red",
                        marker="X",
                        s=80,
                        label="Médias (μ_k)",
                    )

        # Rótulos dos eixos
        def _fname(idx):
            if feature_names is not None and 0 <= idx < len(feature_names):
                return feature_names[idx]
            return f"Feature {idx}"

        ax.set_title(f"Métrica: {metric_name}", fontsize=10)
        ax.set_xlabel(_fname(x_index))
        ax.set_ylabel(_fname(y_index))
        ax.set_zlabel(_fname(z_index))

        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)

        # Agora sim pega tudo (clusters + centros) e mostra legenda
        handles, labels_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")

    # Apaga subplots sobrando, se houver
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"{algorithm_name} — Comparação entre métricas de distância (3D)",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()


def plot_melhor_k_silhouette_and_sse(df_k_search):
    plt.figure(figsize=(12, 5))

    # --- Curva do cotovelo (SSE) ---
    plt.subplot(1, 2, 1)
    plt.plot(df_k_search["k"], df_k_search["sse_euclidean"], marker="o")
    plt.title("Método do cotovelo (SSE)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("SSE (distância euclidiana ao quadrado)")
    plt.grid(True)

    # --- Curva do Silhouette ---
    plt.subplot(1, 2, 2)
    plt.plot(df_k_search["k"], df_k_search["silhouette"], marker="o")
    plt.title("Silhouette médio por k")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def to_probability_simplex(X):
    """
    Transforma cada linha de X em uma distribuição de probabilidade:
    - garante não negatividade
    - normaliza para soma = 1
    """
    X = np.asarray(X, dtype=float)
    X = np.maximum(X, 0.0)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return X / row_sums

