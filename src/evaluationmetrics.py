import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_score,
    completeness_score,
    fowlkes_mallows_score,
)
from scipy.spatial.distance import cdist


class EvaluationMetrics:
    """
    Classe para avaliação de resultados de clusterização.
    Inclui métricas internas (sem rótulos verdadeiros) e externas (quando y_true é conhecido).
    """

    def __init__(self, X, labels, y_true=None):
        """
        Inicializa a classe com os dados e rótulos de cluster.

        :param X: array-like, shape (n_samples, n_features)
            Dados de entrada utilizados na clusterização.
        :param labels: array-like, shape (n_samples,)
            Rótulos atribuídos pelo algoritmo de clusterização.
        :param y_true: array-like, shape (n_samples,), opcional
            Rótulos verdadeiros (se disponíveis).
        """
        self.X = np.asarray(X)
        self.labels = np.asarray(labels)
        self.y_true = np.asarray(y_true) if y_true is not None else None

    # ---------------------- MÉTRICAS INTERNAS ----------------------

    def silhouette(self):
        """Retorna o coeficiente de silhueta (quanto maior, melhor)."""
        if len(set(self.labels)) < 2:
            return np.nan
        return silhouette_score(self.X, self.labels)

    def davies_bouldin(self):
        """Retorna o índice Davies-Bouldin (quanto menor, melhor)."""
        if len(set(self.labels)) < 2:
            return np.nan
        return davies_bouldin_score(self.X, self.labels)

    def calinski_harabasz(self):
        """Retorna o índice Calinski-Harabasz (quanto maior, melhor)."""
        if len(set(self.labels)) < 2:
            return np.nan
        return calinski_harabasz_score(self.X, self.labels)
    
    def sse_euclidean(self):
        """
        Calcula a Soma dos Erros Quadráticos Euclidianos (SSE) entre
        cada ponto e o centróide do seu cluster.
        """

        sse = 0.0
        unique_labels = [k for k in np.unique(self.labels) if k != -1]  # ignora ruído se houver

        for k in unique_labels:
            cluster_points = self.X[self.labels == k]
            if cluster_points.shape[0] == 0:
                continue
            centroid = cluster_points.mean(axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)

        return sse
    # ---------------------- MÉTRICAS EXTERNAS ----------------------

    def homogeneity(self):
        """Homogeneidade (quanto maior, melhor)."""
        if self.y_true is None:
            return np.nan
        return homogeneity_score(self.y_true, self.labels)

    def completeness(self):
        """Completude (quanto maior, melhor)."""
        if self.y_true is None:
            return np.nan
        return completeness_score(self.y_true, self.labels)

    def fowlkes_mallows(self):
        """Índice Fowlkes-Mallows (quanto maior, melhor)."""
        if self.y_true is None:
            return np.nan
        return fowlkes_mallows_score(self.y_true, self.labels)

    # ---------------------- RESUMO GERAL ----------------------

    def summary(self):
        """
        Retorna um dicionário com as principais métricas calculadas.
        Útil para comparação entre algoritmos.
        """
        metrics = {
            "Silhouette": self.silhouette(),
            "Davies-Bouldin": self.davies_bouldin(),
            "Calinski-Harabasz": self.calinski_harabasz()
        }

        # só calcula externas se y_true for fornecido
        if self.y_true is not None:
            metrics.update({
                "Homogeneity": self.homogeneity(),
                "Completeness": self.completeness(),
                "Fowlkes-Mallows": self.fowlkes_mallows(),
            })

        return metrics

    def print_summary(self):
        """Imprime o resumo formatado."""
        metrics = self.summary()
        print("\n Avaliação de Clusterização:")
        for k, v in metrics.items():
            print(f" - {k:<20}: {v:.4f}" if not np.isnan(v) else f" - {k:<20}: N/A")
