import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import DistanceMetrics


class KMeans:
    def __init__(
        self,
        k: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        metric_name: str = "euclidean_distance",
    ):
        """
        Implementação do algoritmo K-Means do zero, com suporte a múltiplas métricas de distância.

        Parâmetros:
        -----------
        k : int
            Número de clusters.
        max_iterations : int
            Número máximo de iterações antes de parar.
        tol : float
            Tolerância para critério de convergência (mudança mínima nos centróides).
        random_state : int, opcional
            Semente aleatória para reprodutibilidade.
        metric_name : str
            Nome da métrica de distância a ser utilizada. Deve estar definida em DistanceMetrics.
            Exemplos: 'euclidean_distance', 'manhattan_distance', 'cosine_distance',
            'mahalanobis_distance', 'entropy_distance', etc.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.metric_name = metric_name

        # Será configurada no fit, pois Mahalanobis depende dos dados
        self.distance_metric = None

        self.centroids: np.ndarray | None = None
        self.sse: float = 0.0
        self.sse_history: list[float] = []

    # -------------------------------------------------------------------------
    def _setup_distance_metric(self, data: np.ndarray) -> None:
        """
        Configura a função de distância com base em metric_name.
        Para Mahalanobis, calcula a matriz de covariância dos dados e fixa inv_cov.
        """
        if self.metric_name == "mahalanobis_distance":
            # Covariância global dos dados
            cov = np.cov(data, rowvar=False)
            # Regularização leve para garantir inversa estável
            cov += 1e-6 * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov)

            # Usa diretamente a função mahalanobis(point1, point2, inv_cov)
            def maha_wrapper(x, y, ic=inv_cov):
                return DistanceMetrics.mahalanobis(x, y, ic)

            self.distance_metric = maha_wrapper
        else:
            # Demais métricas já funcionam com assinatura (x, y)
            self.distance_metric = DistanceMetrics.get_metric(self.metric_name)

    # -------------------------------------------------------------------------
    def initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """
        Inicializa os centróides aleatoriamente escolhendo k pontos únicos.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        else:
            np.random.seed()

        random_idx = np.random.permutation(data.shape[0])
        centroids = data[random_idx[: self.k]]
        return centroids

    # -------------------------------------------------------------------------
    def assign_points_to_clusters(self, data: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto ao cluster do centróide mais próximo usando a métrica selecionada.
        """
        if self.distance_metric is None:
            raise ValueError(
                "distance_metric não foi inicializada. "
                "Certifique-se de chamar fit(data) antes de assign_points_to_clusters."
            )

        distances = np.zeros((data.shape[0], self.k))

        for i, point in enumerate(data):
            for j, centroid in enumerate(self.centroids):
                distances[i, j] = self.distance_metric(point, centroid)

        return np.argmin(distances, axis=1)

    # -------------------------------------------------------------------------
    def update_centroids(self, data: np.ndarray, clusters: np.ndarray) -> np.ndarray:
            """
            Atualiza os centróides.
            - Usa a MEDIANA se a métrica for Manhattan.
            - Usa a MÉDIA para as outras métricas (Euclidiana = exato; JSD/Cosseno/Mahalanobis = heurística).
            Reposiciona o centróide aleatoriamente caso o cluster fique vazio.
            """
            new_centroids = []
            for k_idx in range(self.k):
                cluster_points = data[clusters == k_idx]
                
                if len(cluster_points) == 0:
                    # evita centróide "vazio"
                    new_centroids.append(data[np.random.randint(0, data.shape[0])])
                else:
                    # --- AQUI ESTÁ A MUDANÇA ---
                    if self.metric_name == 'manhattan_distance':
                        # Para Manhattan, a Mediana minimiza o erro
                        new_centroids.append(np.median(cluster_points, axis=0))
                    else:
                        # Para Euclidiana (e heurística das demais), usa-se a Média
                        new_centroids.append(cluster_points.mean(axis=0))
                        
            return np.array(new_centroids)

    # -------------------------------------------------------------------------
    def calculate_sse(self, data: np.ndarray, clusters: np.ndarray) -> float:
        """
        Calcula o SSE (Soma dos Erros Quadráticos) após a formação dos clusters.

        IMPORTANTE:
        -----------
        O SSE aqui é calculado sempre com base na distância Euclidiana entre cada
        ponto e o centróide do seu cluster, independentemente da métrica utilizada
        para formar os clusters (euclidiana, manhattan, mahalanobis, etc.). Isso
        permite comparar a "compacidade" dos clusters em um espaço comum.
        """
        # Distância Euclidiana ao quadrado (norma L2) entre ponto e centróide
        diff = data - self.centroids[clusters]
        return float(np.sum(diff**2))

    # -------------------------------------------------------------------------
    def fit(self, data: np.ndarray):
        """
        Ajusta o modelo aos dados.
        Reatribui clusters e atualiza centróides até convergir ou atingir o limite de iterações.
        """
        data = np.asarray(data, dtype=float)

        # Configura a métrica (para Mahalanobis, depende dos dados)
        self._setup_distance_metric(data)

        # Inicializa centróides
        self.centroids = self.initialize_centroids(data)
        self.sse_history.clear()

        for i in range(self.max_iterations):
            clusters = self.assign_points_to_clusters(data)
            new_centroids = self.update_centroids(data, clusters)

            sse = self.calculate_sse(data, clusters)
            self.sse_history.append(sse)
            self.sse = sse

            # Critério de convergência com tolerância
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                print(f"Convergiu na iteração {i + 1}")
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        return self.centroids, clusters

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
        Exibe a distribuição dos pontos e centróides em 2D.
        """
        data = np.asarray(data, dtype=float)

        if data.shape[1] != 2:
            raise ValueError("O plot de clusters está disponível apenas para dados 2D.")

        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis", s=30)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="red", marker="X", s=200)
        plt.title(f"K-Means com {self.k} clusters ({self.metric_name})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    # -------------------------------------------------------------------------
    def plot_sse(self) -> None:
        """
        Plota a evolução do SSE ao longo das iterações.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.sse_history) + 1), self.sse_history, "o-")
        plt.title("Evolução do SSE por Iteração")
        plt.xlabel("Iteração")
        plt.ylabel("SSE (distância Euclidiana ao quadrado)")
        plt.grid(True)
        plt.show()
