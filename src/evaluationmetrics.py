import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_score,
    completeness_score,
    fowlkes_mallows_score,
)
from scipy.spatial.distance import cdist, jensenshannon

class EvaluationMetrics:
    # ... (outros imports)

    def __init__(self, X, labels, y_true=None, metric='euclidean', **kwargs):
        """
        :param metric: A métrica (str ou callable).
        :param kwargs: Parâmetros extras para a métrica (ex: V=VI para Mahalanobis).
        """
        self.X = np.asarray(X)
        self.labels = np.asarray(labels)
        self.y_true = np.asarray(y_true) if y_true is not None else None
        self.metric = metric
        self.metric_params = kwargs  # <--- Guardamos os parâmetros extras aqui

    def silhouette(self):
        if len(set(self.labels)) < 2:
            return np.nan
        try:
            # Repassamos os parâmetros extras (como a matriz VI) para o score
            return silhouette_score(self.X, self.labels, metric=self.metric, **self.metric_params)
        except Exception as e:
            print(f"Erro Silhouette: {e}")
            return np.nan

    def davies_bouldin(self):
        """
        Retorna o índice Davies-Bouldin.
        Nota: O sklearn calcula isso sempre com distância Euclidiana por padrão.
        """
        if len(set(self.labels)) < 2:
            return np.nan
        return davies_bouldin_score(self.X, self.labels)

    def calinski_harabasz(self):
        """
        Retorna o índice Calinski-Harabasz.
        Nota: O sklearn calcula isso sempre com distância Euclidiana por padrão.
        """
        if len(set(self.labels)) < 2:
            return np.nan
        return calinski_harabasz_score(self.X, self.labels)
    
    def sse_euclidean(self):
        """Calcula SSE (sempre Euclidiano pois depende do centróide médio)."""
        sse = 0.0
        unique_labels = [k for k in np.unique(self.labels) if k != -1]

        for k in unique_labels:
            cluster_points = self.X[self.labels == k]
            if cluster_points.shape[0] == 0:
                continue
            centroid = cluster_points.mean(axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)

        return sse

    # ---------------------- MÉTRICAS EXTERNAS ----------------------
    # (Mantive igual, pois não dependem de distância, só dos rótulos)
    
    def homogeneity(self):
        if self.y_true is None: return np.nan
        return homogeneity_score(self.y_true, self.labels)

    def completeness(self):
        if self.y_true is None: return np.nan
        return completeness_score(self.y_true, self.labels)

    def fowlkes_mallows(self):
        if self.y_true is None: return np.nan
        return fowlkes_mallows_score(self.y_true, self.labels)

    # ---------------------- RESUMO ----------------------

    def summary(self):
        metrics = {
            "Silhouette": self.silhouette(),
            "Davies-Bouldin": self.davies_bouldin(),
            "Calinski-Harabasz": self.calinski_harabasz()
        }
        if self.y_true is not None:
            metrics.update({
                "Homogeneity": self.homogeneity(),
                "Completeness": self.completeness(),
                "Fowlkes-Mallows": self.fowlkes_mallows(),
            })
        return metrics

    def print_summary(self):
        metrics = self.summary()
        print(f"\n Avaliação de Clusterização (Métrica: {self.metric}):")
        for k, v in metrics.items():
            print(f" - {k:<20}: {v:.4f}" if not np.isnan(v) else f" - {k:<20}: N/A")