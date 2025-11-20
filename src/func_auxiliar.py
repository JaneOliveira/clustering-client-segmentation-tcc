#Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os
import sys
import pandas as pd
from time import time

from sklearn.datasets import make_blobs

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

def calculate_clustering_metrics(algorithm_name, metric_name, runtime, X, labels, eval_metric='euclidean', eval_kwargs=None):
    """
    Calcula as métricas de clusterização.
    Agora aceita 'eval_metric' e 'eval_kwargs' para calcular o Silhouette correto.
    """
    if eval_kwargs is None:
        eval_kwargs = {}

    # --- AQUI ESTÁ A CORREÇÃO ---
    # Repassamos a métrica (eval_metric) e os argumentos extras (eval_kwargs) para a classe
    evaluator = EvaluationMetrics(X, labels, y_true=None, metric=eval_metric, **eval_kwargs)
    
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
