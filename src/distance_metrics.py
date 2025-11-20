import numpy as np

class DistanceMetrics:
    """
    Coleção de métricas de distância utilizadas neste trabalho.
    - Euclidiana
    - Manhattan
    - Cosseno
    - Mahalanobis
    - Jensen–Shannon Distance (JSD)
    """

    # -------------------------------------------------------------
    # Distâncias Clássicas
    # -------------------------------------------------------------
    @staticmethod
    def euclidean(point1, point2):
        """Distância Euclidiana."""
        # Euclidiana: ||x−y||₂
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def manhattan(point1, point2):
        # Manhattan: ||x−y||₁
        """Distância Manhattan (L1)."""
        return np.sum(np.abs(point1 - point2))

    @staticmethod
    def cosine(point1, point2):
        """Distância do Cosseno (1 - cos)."""
        # Distância do Cosseno: 1 - (x·y) / (||x|| ||y||)
        dot = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1 - dot / (norm1 * norm2 + 1e-12)

    # -------------------------------------------------------------
    # Mahalanobis — duas versões
    # -------------------------------------------------------------
    @staticmethod
    def mahalanobis(point1, point2, inv_cov):
        """
        Cálculo direto da distância de Mahalanobis:
        d = sqrt( (x - μ)^T Σ^{-1} (x - μ) )
        """
        # Mahalanobis: sqrt( (x - μ)^T Σ^{-1} (x - μ) )
        diff = point1 - point2
        return float(np.sqrt(diff.T @ inv_cov @ diff))

    @staticmethod
    def mahalanobis_wrapper(inv_cov):
        """
        Retorna uma função de distância Mahalanobis fixando Σ⁻¹.
        Usado para integração simples com algoritmos de cluster.
        """
        def _dist(p1, p2):
            diff = p1 - p2
            return float(np.sqrt(diff.T @ inv_cov @ diff))
        return _dist

    # -------------------------------------------------------------
    # Jensen–Shannon Distance (JSD)
    # -------------------------------------------------------------
    @staticmethod
    def jensen_shannon(point1, point2, base=np.e):
        """
        Jensen–Shannon Distance = sqrt(JS Divergence).

        É definida como:
            JSD(p, q) = sqrt( 0.5 * KL(p||m) + 0.5 * KL(q||m) )
        com m = (p+q)/2

        Requer vetores normalizados como distribuições.
        """
        # normalização para comportamento probabilístico
        p = np.abs(point1) / (np.sum(np.abs(point1)) + 1e-12)
        q = np.abs(point2) / (np.sum(np.abs(point2)) + 1e-12)
        m = 0.5 * (p + q)

        log = lambda x: np.log(x) / np.log(base)

        def kl(a, b):
            a = np.clip(a, 1e-12, 1)
            b = np.clip(b, 1e-12, 1)
            return np.sum(a * (log(a) - log(b)))

        js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return np.sqrt(js_div)

    # -------------------------------------------------------------
    # Mapeamento de nomes → funções
    # -------------------------------------------------------------
    @staticmethod
    def get_metric(name, inv_cov=None):
        """
        Retorna uma função de distância:
        - Para Mahalanobis, pode receber Σ⁻¹ opcional.
        - Para demais métricas, basta o nome.
        """
        name = name.lower().strip()

        metrics = {
            "euclidean_distance": DistanceMetrics.euclidean,
            "manhattan_distance": DistanceMetrics.manhattan,
            "cosine_distance": DistanceMetrics.cosine,
            "jensen-shannon_distance": DistanceMetrics.jensen_shannon,
        }

        if name == "mahalanobis_distance":
            if inv_cov is None:
                raise ValueError("Mahalanobis requer inv_cov_matrix.")
            return DistanceMetrics.mahalanobis_wrapper(inv_cov)

        return metrics.get(name, DistanceMetrics.euclidean)
