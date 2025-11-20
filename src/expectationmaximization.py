import warnings
import numpy as np
import matplotlib.pyplot as plt
from distance_metrics import DistanceMetrics


class ExpectationMaximization:
    """
    Algoritmo de Clusterização Expectation-Maximization (EM) para Gaussian Mixture Models (GMM),
    com suporte a uma variante experimental baseada em métricas de distância arbitrárias.

    Modo clássico (GMM correto):
        - metric_name = 'mahalanobis_distance'
        - Usa densidade Gaussiana multivariada com covariâncias por cluster.
        - Interpretação probabilística válida, log-verossimilhança verdadeira.

    Modo experimental (soft clustering baseado em distância):
        - metric_name != 'mahalanobis_distance'
        - Usa kernel do tipo exp(-0.5 * d(x, μ_k)^2) sem normalização Gaussiana.
        - Não é mais um GMM formal; log-verossimilhança vira um pseudo-score.

    Importante sobre SSE:
        - Além da log-verossimilhança (ou pseudo-log-likelihood), este modelo calcula
          também um SSE baseado SEMPRE na distância Euclidiana entre cada ponto e a
          média (μ_k) do cluster ao qual foi atribuído.
        - Isso permite comparar o EM, de forma geométrica, com outros algoritmos
          (K-Means, K-Medoids, Ward, DBSCAN), independentemente da métrica usada
          internamente no EM.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        metric_name: str = "mahalanobis_distance",
        p: int = 3,
    ):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.metric_name = metric_name
        self.p = p  # se precisar em alguma métrica Lp

        # Parâmetros aprendidos (modo GMM)
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

        # Histórico de log-verossimilhança (ou pseudo-log-likelihood no modo experimental)
        self.log_likelihood_: list[float] = []

        # Responsabilidades e rótulos finais
        self.responsibilities_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

        # SSE Euclidiano para comparação geométrica
        self.sse_euclidean_: float | None = None
        self.sse_euclidean_history_: list[float] = []

        # Métrica de distância (apenas para modo experimental)
        if self.metric_name == "mahalanobis_distance":
            # No modo GMM clássico, a "distância" já está embutida na Gaussiana
            self.metric_func = None
        else:
            self.metric_func = DistanceMetrics.get_metric(metric_name)
            warnings.warn(
                "ExpectationMaximization: com métricas diferentes de "
                "'mahalanobis_distance', o algoritmo deixa de ser um GMM "
                "probabilisticamente consistente e passa a ser uma variante "
                "experimental de soft-clustering baseada em distância.",
                UserWarning,
            )

    # ------------------------------------------------------------------ #
    # Inicialização
    # ------------------------------------------------------------------ #
    def initialize_parameters(self, X: np.ndarray) -> None:
        """Inicializa médias, covariâncias e pesos de mistura."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)

        # Médias iniciais
        self.means_ = X[random_idx]

        # Covariância inicial: mesma matriz global para todos, com regularização
        global_cov = np.cov(X, rowvar=False)
        eps = 1e-6 * np.eye(n_features)  # regularização para evitar singularidade
        global_cov = global_cov + eps

        self.covariances_ = np.array([global_cov.copy() for _ in range(self.n_clusters)])

        # Pesos iniciais uniformes
        self.weights_ = np.ones(self.n_clusters) / self.n_clusters

    # ------------------------------------------------------------------ #
    # Densidade Gaussiana
    # ------------------------------------------------------------------ #
    def gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Calcula a densidade da distribuição Gaussiana multivariada para cada ponto em X.
        Usada apenas no modo GMM clássico (metric_name='mahalanobis_distance').
        """
        n_features = X.shape[1]
        det_cov = np.linalg.det(cov)

        # Regularização extra se cov estiver degenerada
        if det_cov <= 0:
            cov = cov + 1e-6 * np.eye(n_features)
            det_cov = np.linalg.det(cov)

        inv_cov = np.linalg.inv(cov)
        norm_factor = 1.0 / np.sqrt((2 * np.pi) ** n_features * det_cov)

        diff = X - mean
        # (x - μ)^T Σ^{-1} (x - μ) para cada linha de X
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

        return norm_factor * np.exp(exponent)

    # ------------------------------------------------------------------ #
    # Passo E
    # ------------------------------------------------------------------ #
    def expectation_step(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula as responsabilidades γ_{nk} = P(z_k | x_n).

        - Modo GMM clássico (Mahalanobis): usa densidade Gaussiana verdadeira.
        - Modo experimental: usa kernel exp(-0.5 * d(x, μ_k)^2) como "score".
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))

        if self.metric_name == "mahalanobis_distance":
            # GMM clássico: densidade Gaussiana multivariada
            for k in range(self.n_clusters):
                pdf_k = self.gaussian_pdf(X, self.means_[k], self.covariances_[k])
                responsibilities[:, k] = self.weights_[k] * pdf_k
        else:
            # Modo experimental: usa kernel baseado em distância
            for k in range(self.n_clusters):
                distances = np.array([self.metric_func(x, self.means_[k]) for x in X])
                score = np.exp(-0.5 * distances**2)
                responsibilities[:, k] = self.weights_[k] * score

        # Normaliza para obter "probabilidades" (ou pseudo-probabilidades)
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12  # evitar divisão por zero
        responsibilities /= row_sums

        return responsibilities

    # ------------------------------------------------------------------ #
    # Passo M
    # ------------------------------------------------------------------ #
    def maximization_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        Atualiza pesos, médias e covariâncias com base nas responsabilidades.
        Essa atualização é consistente com o GMM clássico. No modo experimental,
        ela ainda é utilizada para obter centróides e matrizes de dispersão,
        mas sem interpretação probabilística estrita.
        """
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)  # soma das responsabilidades por cluster

        # Atualiza pesos
        self.weights_ = Nk / n_samples

        # Atualiza médias
        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

        # Atualiza covariâncias
        self.covariances_ = np.zeros((self.n_clusters, n_features, n_features))

        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k][:, np.newaxis] * diff
            cov_k = weighted_diff.T @ diff / Nk[k]

            # Regularização para estabilidade numérica
            cov_k += 1e-6 * np.eye(n_features)
            self.covariances_[k] = cov_k

    # ------------------------------------------------------------------ #
    # Log-verossimilhança
    # ------------------------------------------------------------------ #
    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Calcula a log-verossimilhança total do modelo.

        - Modo Mahalanobis: log-verossimilhança verdadeira de um GMM.
        - Modo experimental: log de um "pseudo-likelihood" baseado no kernel
          exp(-0.5 d(x, μ_k)^2), usado apenas como critério relativo de convergência.
        """
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.n_clusters))

        if self.metric_name == "mahalanobis_distance":
            # GMM clássico
            for k in range(self.n_clusters):
                pdf_k = self.gaussian_pdf(X, self.means_[k], self.covariances_[k])
                likelihood[:, k] = self.weights_[k] * pdf_k
        else:
            # Modo experimental: mesmas "densidades" do passo E
            for k in range(self.n_clusters):
                distances = np.array([self.metric_func(x, self.means_[k]) for x in X])
                score = np.exp(-0.5 * distances**2)
                likelihood[:, k] = self.weights_[k] * score

        row_sums = likelihood.sum(axis=1)
        row_sums[row_sums <= 1e-300] = 1e-300  # evitar log(0)

        return float(np.sum(np.log(row_sums)))

    # ------------------------------------------------------------------ #
    # SSE Euclidiano (geométrico)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_sse_euclidean(X: np.ndarray, labels: np.ndarray, means: np.ndarray) -> float:
        """
        Calcula o SSE Euclidiano: soma das distâncias quadráticas de cada ponto
        até a média do cluster ao qual foi atribuído.

        Este SSE é sempre baseado na distância Euclidiana, independentemente da
        métrica usada internamente no EM. Serve para comparar geometricamente
        com K-Means, K-Medoids, Ward e DBSCAN.
        """
        sse = 0.0
        for k in range(means.shape[0]):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] == 0:
                continue
            diff = cluster_points - means[k]
            sse += float(np.sum(diff ** 2))
        return sse

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray):
        """
        Treina o modelo EM nos dados.

        - X: array (n_samples, n_features)

        Ao final, além da log-verossimilhança, é calculado também o SSE
        Euclidiano com base nos rótulos finais e nas médias μ_k estimadas.
        """
        X = np.asarray(X, dtype=float)
        self.initialize_parameters(X)
        self.log_likelihood_.clear()
        self.sse_euclidean_history_.clear()

        prev_log_likelihood = None

        for i in range(self.max_iterations):
            # Passo E
            responsibilities = self.expectation_step(X)

            # Passo M
            self.maximization_step(X, responsibilities)

            # Log-verossimilhança
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihood_.append(log_likelihood)

            # Critério de parada
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    break

            prev_log_likelihood = log_likelihood

        # Guarda responsabilidades e rótulos finais (hard assignment)
        self.responsibilities_ = responsibilities
        self.labels_ = np.argmax(responsibilities, axis=1)

        # Calcula SSE Euclidiano para comparação geométrica
        self.sse_euclidean_ = self._compute_sse_euclidean(X, self.labels_, self.means_)
        self.sse_euclidean_history_.append(self.sse_euclidean_)

        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna o cluster mais provável (ou mais provável no sentido experimental) para cada ponto.
        """
        X = np.asarray(X, dtype=float)
        responsibilities = self.expectation_step(X)
        return np.argmax(responsibilities, axis=1)

    # ------------------------------------------------------------------ #
    # Plots auxiliares
    # ------------------------------------------------------------------ #
    def plot_clusters(self, X: np.ndarray, figsize=(6, 6)) -> None:
        """
        Plota os clusters e suas médias para dados 2D.
        """
        if self.labels_ is None:
            raise ValueError("Execute fit(X) antes de plotar os clusters.")

        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("plot_clusters espera dados 2D (duas features).")

        plt.figure(figsize=figsize)
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=self.labels_,
            cmap="viridis",
            s=50,
            edgecolors="k",
            alpha=0.8,
        )
        if self.means_ is not None:
            plt.scatter(
                self.means_[:, 0],
                self.means_[:, 1],
                c="red",
                marker="x",
                s=150,
                label="Médias (μ_k)",
            )
        plt.title(f"Clusters - EM ({self.metric_name})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_log_likelihood(self) -> None:
        """
        Plota a evolução da log-verossimilhança (ou pseudo-log-likelihood).
        """
        if len(self.log_likelihood_) == 0:
            raise ValueError("Modelo ainda não foi ajustado com fit(X).")

        plt.figure(figsize=(7, 4))
        plt.plot(self.log_likelihood_, marker="o")
        plt.title("Convergência do EM - Log-Verossimilhança")
        plt.xlabel("Iterações")
        plt.ylabel("Log-Verossimilhança")
        plt.grid(True)
        plt.show()

    def plot_sse_euclidean(self) -> None:
        """
        Plota o SSE Euclidiano armazenado (normalmente um único ponto,
        já que o EM não é iterativo em SSE, mas em log-likelihood).

        Foi mantido como função separada para manter consistência com
        os demais algoritmos do trabalho (K-Means, K-Medoids, Ward, DBSCAN).
        """
        if len(self.sse_euclidean_history_) == 0:
            raise ValueError("O modelo ainda não foi ajustado com fit(X).")

        plt.figure(figsize=(7, 4))
        plt.plot(self.sse_euclidean_history_, marker="o")
        plt.title("SSE Euclidiano - EM")
        plt.xlabel("Execuções / Experimentos")
        plt.ylabel("SSE (distância Euclidiana)")
        plt.grid(True)
        plt.show()
