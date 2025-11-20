import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import DistanceMetrics


class DBSCAN:
    """
    Implementação manual do algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Parâmetros
    ----------
    eps : float
        Raio máximo para considerar dois pontos como vizinhos.
    min_samples : int
        Número mínimo de pontos necessários para formar um cluster denso.
    metric : str
        Nome da métrica de distância (usa DistanceMetrics).
        Ex.: 'euclidean_distance', 'manhattan_distance', 'cosine_distance',
        'mahalanobis_distance', 'jensen-shannon_distance'.
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean_distance"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric_name = metric

        # será configurada no fit (especialmente para Mahalanobis)
        self.distance_func = None
        self.inv_cov = None

        # Atributos de saída
        self.labels_ = None          # rótulos: -1 = ruído, 0 = não usado, >0 = id cluster
        self.n_clusters_ = 0
        self.noise_ = None
        self.core_samples_indices_ = None
        self.sse_ = None
        self.sse_history = []

    # -------------------------------------------------------------------------
    def _setup_metric(self, X: np.ndarray) -> None:
        """
        Configura self.distance_func considerando a métrica escolhida.
        Para Mahalanobis, calcula uma matriz de covariância global e cria
        um wrapper que injeta inv_cov automaticamente.
        """
        if self.metric_name == "mahalanobis_distance":
            # Matriz de covariância global dos dados
            cov = np.cov(X, rowvar=False)
            cov += 1e-6 * np.eye(cov.shape[1])  # regularização
            self.inv_cov = np.linalg.inv(cov)

            # Wrapper que usa a Mahalanobis fixa
            def maha_wrapper(p1, p2, ic=self.inv_cov):
                return DistanceMetrics.mahalanobis(p1, p2, ic)

            self.distance_func = maha_wrapper
        else:
            # Outras métricas: assumimos assinatura f(x, y)
            self.distance_func = DistanceMetrics.get_metric(self.metric_name)

    # -------------------------------------------------------------------------
    def _region_query(self, X, point_idx):
        """Retorna os índices dos vizinhos de X[point_idx] dentro de eps."""
        neighbors = []
        for j in range(X.shape[0]):
            if self.distance_func(X[point_idx], X[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    # -------------------------------------------------------------------------
    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expande o cluster a partir de um ponto core.

        labels: array de rótulos (modificado in-place)
        neighbors: lista de índices vizinhos do ponto inicial
        """
        # rótulos: 0 = UNVISITED, -1 = NOISE, >0 = cluster
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # Se estava marcado como ruído, vira ponto de borda associado ao cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            # Se ainda não foi visitado, visita e tenta expandir
            if labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            i += 1

    # -------------------------------------------------------------------------
    def fit(self, X):
        """
        Executa o algoritmo DBSCAN para identificar clusters densos e ruído.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Configura função de distância (incluindo Mahalanobis, se for o caso)
        self._setup_metric(X)

        # 0 = não visitado, -1 = ruído, >0 = id do cluster
        labels = np.zeros(n, dtype=int)
        cluster_id = 0

        for i in range(n):
            # Se já visitado (ruído ou cluster), pula
            if labels[i] != 0:
                continue

            neighbors = self._region_query(X, i)

            # Não é ponto core
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # ruído
            else:
                # Novo cluster
                cluster_id += 1
                self._expand_cluster(X, labels, i, neighbors, cluster_id)

        # Finalizações
        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.noise_ = np.sum(labels == -1)

        # Core samples: pontos não ruído com pelo menos min_samples vizinhos
        core_indices = []
        for i in range(n):
            if labels[i] == -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) >= self.min_samples:
                core_indices.append(i)
        self.core_samples_indices_ = np.array(core_indices, dtype=int)

        # Calcula SSE intra-cluster (sempre Euclidiano, para comparação com outros métodos)
        self.sse_ = self._calculate_sse(X, labels)
        self.sse_history.append(self.sse_)

        return self

    # -------------------------------------------------------------------------
    def fit_predict(self, X):
        """Atalho: ajusta o modelo e retorna os rótulos dos clusters."""
        self.fit(X)
        return self.labels_

    # -------------------------------------------------------------------------
    def _calculate_sse(self, X, labels):
        """
        Calcula a soma das distâncias quadráticas intra-cluster (SSE).
        Observação: aqui é usada sempre a distância Euclidiana,
        independentemente da métrica de vizinhança do DBSCAN, para manter
        comparabilidade com os demais algoritmos.
        """
        sse = 0.0
        unique_clusters = set(labels)
        for cluster_id in unique_clusters:
            if cluster_id <= 0:
                continue  # ignora ruído e rótulo 0 (que não deve sobrar)
            cluster_points = X[labels == cluster_id]
            if cluster_points.shape[0] == 0:
                continue
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse

    # -------------------------------------------------------------------------
    def plot_clusters(self, X, figsize=(7, 6)):
        """Plota os clusters encontrados e marca os pontos de ruído."""
        if self.labels_ is None:
            raise ValueError("Execute fit() antes de plotar os clusters.")

        X = np.asarray(X)
        plt.figure(figsize=figsize)
        unique_labels = set(self.labels_)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for k, color in zip(unique_labels, colors):
            if k == -1:
                color = "k"  # ruído em preto
                label = "Ruído"
            else:
                label = f"Cluster {k}"
            mask = self.labels_ == k
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                s=60,
                c=[color],
                label=label,
                edgecolors="k",
            )

        plt.title(
            f"DBSCAN - Métrica interna: {self.metric_name} | "
            f"Clusters: {self.n_clusters_} | Ruído: {self.noise_}"
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------------------------
    def plot_sse(self):
        """Plota o SSE total do DBSCAN."""
        if len(self.sse_history) == 0:
            raise ValueError("O modelo ainda não foi ajustado.")
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(self.sse_history) + 1),
            self.sse_history,
            marker="o",
        )
        plt.title("Soma das Distâncias Quadráticas Intra-Cluster (SSE) - DBSCAN")
        plt.xlabel("Iterações (sempre 1 para DBSCAN)")
        plt.ylabel("SSE Total (Euclidiano)")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------------------------
    def plot_k_distance(self, X, k=None, sample_size=500):
        """
        Gera o gráfico de distância ao k-ésimo vizinho (k-distance plot),
        utilizado para estimar o valor ideal de eps no DBSCAN.

        Parâmetros
        ----------
        X : array-like
            Dados de entrada.
        k : int | None
            Ordem do vizinho considerado. Se None, usa min_samples.
        sample_size : int
            Número máximo de pontos amostrados (para performance).
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        k = k or self.min_samples

        # Reduz amostra para agilizar (mantendo qualidade)
        if n > sample_size:
            rng = np.random.default_rng(42)
            X = rng.choice(X, size=sample_size, replace=False)

        # Garante que a métrica esteja configurada
        self._setup_metric(X)

        # Calcula distâncias de todos para todos (usando a métrica escolhida)
        m = X.shape[0]
        dist_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                dist = self.distance_func(X[i], X[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        # Ordena as distâncias de cada ponto e pega o k-ésimo vizinho
        sorted_dists = np.sort(dist_matrix, axis=1)
        k_dists = sorted_dists[:, k]

        # Ordena todas as distâncias para o gráfico do cotovelo
        k_dists_sorted = np.sort(k_dists)

        # Plot do k-distance
        plt.figure(figsize=(8, 5))
        plt.plot(k_dists_sorted, linewidth=2)
        plt.title(
            f"Distância ao {k}-ésimo Vizinho - Método do Cotovelo para eps (DBSCAN)\n"
            f"Métrica interna: {self.metric_name}"
        )
        plt.xlabel("Pontos ordenados")
        plt.ylabel(f"Distância até o {k}-ésimo vizinho")
        plt.grid(True)
        plt.show()

        print("Use o ponto de inflexão (cotovelo) para escolher o eps ideal.")
