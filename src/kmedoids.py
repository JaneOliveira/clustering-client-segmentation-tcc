import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import DistanceMetrics


class KMedoids:
    def __init__(
        self,
        k: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        metric_name: str = "euclidean_distance",
    ):
        """
        Implementação do algoritmo K-Medoids do zero, com suporte a múltiplas
        métricas de distância via DistanceMetrics.

        Parâmetros
        ----------
        k : int
            Número de clusters.
        max_iterations : int
            Número máximo de iterações antes de parar.
        tol : float
            Tolerância para critério de convergência (mudança mínima nos medoides).
        random_state : int, opcional
            Semente aleatória para reprodutibilidade.
        metric_name : str
            Nome da métrica de distância (deve existir em DistanceMetrics.get_metric()).
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.metric_name = metric_name

        # Base da métrica (pode precisar de inv_cov para Mahalanobis)
        self.distance_metric = None  # será definido no fit, se necessário
        self.inv_cov = None

        self.medoids = None
        self.sse = 0.0
        self.sse_history = []

    # -------------------------------------------------------------------------
    def _setup_metric(self, data: np.ndarray):
        """
        Configura self.distance_metric considerando a métrica escolhida.
        Para Mahalanobis, calcula uma matriz de covariância global e cria
        um wrapper que injeta inv_cov automaticamente.
        """
        if self.metric_name == "mahalanobis_distance":
            # Matriz de covariância global dos dados
            cov = np.cov(data, rowvar=False)
            # Regularização para evitar singularidade
            cov += 1e-6 * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov)

            # Usa diretamente a função mahalanobis(point1, point2, inv_cov)
            def maha_wrapper(x, y, ic=inv_cov):
                return DistanceMetrics.mahalanobis(x, y, ic)
          
            self.distance_metric = maha_wrapper
        else:
            # Para todas as outras métricas, assumimos assinatura f(x, y)
            self.distance_metric = DistanceMetrics.get_metric(self.metric_name)

    # -------------------------------------------------------------------------
    def initialize_medoids(self, data: np.ndarray) -> np.ndarray:
        """
        Inicializa os medoides aleatoriamente escolhendo k pontos únicos do dataset.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed()

        random_idx = np.random.permutation(data.shape[0])
        medoids = data[random_idx[:self.k]]
        return medoids

    # -------------------------------------------------------------------------
    def assign_points_to_clusters(self, data: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto ao medoide mais próximo de acordo com a distância selecionada.
        """
        distances = np.zeros((data.shape[0], self.k))

        for i, point in enumerate(data):
            for j, medoid in enumerate(self.medoids):
                distances[i, j] = self.distance_metric(point, medoid)

        return np.argmin(distances, axis=1)

    # -------------------------------------------------------------------------
    def calculate_sse(self, data: np.ndarray, clusters: np.ndarray) -> float:
        """
        Calcula o SSE (soma das distâncias quadradas de cada ponto ao seu medoide),
        utilizando distância Euclidiana no espaço original.

        Observação:
            Mesmo quando a métrica de clusterização não é Euclidiana,
            este SSE é mantido como uma medida geométrica de erro em R^d,
            útil para comparação entre execuções. Essa escolha é discutida
            no texto do trabalho.
        """
        return float(np.sum((data - self.medoids[clusters]) ** 2))

    # -------------------------------------------------------------------------
    def update_medoids(self, data: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """
        Atualiza os medoides através de uma busca local:
        Para cada cluster, escolhe o ponto cujo custo total (soma das distâncias,
        conforme a métrica selecionada) até os demais pontos do cluster é o menor.
        """
        new_medoids = np.copy(self.medoids)

        for k in range(self.k):
            cluster_points = data[clusters == k]
            if len(cluster_points) == 0:
                # Caso um cluster fique vazio, reatribui medoide aleatoriamente
                new_medoids[k] = data[np.random.randint(0, data.shape[0])]
                continue

            # Matriz de distâncias dentro do cluster usando a métrica escolhida
            n_cluster = cluster_points.shape[0]
            distances = np.zeros((n_cluster, n_cluster))

            for i in range(n_cluster):
                for j in range(n_cluster):
                    if i == j:
                        distances[i, j] = 0.0
                    else:
                        distances[i, j] = self.distance_metric(
                            cluster_points[i], cluster_points[j]
                        )

            total_distances = np.sum(distances, axis=1)

            # Escolhe o ponto com menor soma de distâncias como novo medoide
            new_medoids[k] = cluster_points[np.argmin(total_distances)]

        return new_medoids

    # -------------------------------------------------------------------------
    def fit(self, data: np.ndarray):
        """
        Ajusta o modelo K-Medoids aos dados.
        Reatribui clusters e atualiza medoides até convergir ou atingir o limite de iterações.
        """
        data = np.asarray(data, dtype=float)

        # Configura função de distância (incluindo Mahalanobis, se for o caso)
        self._setup_metric(data)

        self.medoids = self.initialize_medoids(data)
        self.sse_history.clear()

        for i in range(self.max_iterations):
            clusters = self.assign_points_to_clusters(data)
            new_medoids = self.update_medoids(data, clusters)

            sse = self.calculate_sse(data, clusters)
            self.sse_history.append(sse)
            self.sse = sse

            # Critério de convergência
            diff = np.linalg.norm(self.medoids - new_medoids)
            if diff < self.tol:
                print(f"K-Medoids convergiu na iteração {i+1}")
                break

            self.medoids = new_medoids

        return self.medoids, clusters

    # -------------------------------------------------------------------------
    def predict(self, new_data: np.ndarray) -> np.ndarray:
        """
        Prediz o cluster mais próximo para novos pontos.
        """
        new_data = np.asarray(new_data, dtype=float)
        return self.assign_points_to_clusters(new_data)

    # -------------------------------------------------------------------------
    def plot_clusters(self, data: np.ndarray, clusters: np.ndarray) -> None:
        """
        Exibe a distribuição dos pontos e medoides em 2D.
        """
        if data.shape[1] != 2:
            raise ValueError("O plot de clusters está disponível apenas para dados 2D.")

        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis", s=30)
        plt.scatter(
            self.medoids[:, 0],
            self.medoids[:, 1],
            c="red",
            marker="X",
            s=200,
        )
        plt.title(f"K-Medoids com {self.k} clusters - Métrica: {self.metric_name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    # -------------------------------------------------------------------------
    def plot_sse(self) -> None:
        """
        Plota a evolução do SSE (Euclidiano) ao longo das iterações.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.sse_history) + 1), self.sse_history, "o-")
        plt.title("Evolução do SSE por Iteração (K-Medoids)")
        plt.xlabel("Iteração")
        plt.ylabel("SSE (Euclidiano)")
        plt.grid(True)
        plt.show()
