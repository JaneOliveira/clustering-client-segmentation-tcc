import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from distance_metrics import DistanceMetrics


class WardMethod:
    """
    Implementação manual do método hierárquico de Ward (linkage de variância mínima).

    Parâmetros
    ----------
    k : int | None
        Número de clusters desejados. Se None, é necessário usar distance_threshold
        ou apenas inspecionar o dendrograma.
    distance_threshold : float | None
        Alternativa a k: corta a árvore quando o próximo merge ultrapassa esse valor.
        O valor é aplicado sobre o "custo de Ward" (ΔSSE incremental) armazenado em
        linkage[:, 2].
    metric : str
        Nome da métrica de distância (usa DistanceMetrics).
        Tecnicamente, o método de Ward clássico é definido para distância Euclidiana.
        Outras métricas são experimentais e perdem a interpretação exata de
        "variância mínima".
    copy : bool
        Se True, copia X internamente.
    """

    def __init__(self, k=None, distance_threshold=None,
                 metric: str = "euclidean_distance", copy: bool = True):
        if k is not None and distance_threshold is not None:
            raise ValueError("Use apenas k ou distance_threshold, não ambos.")

        self.k = k
        self.distance_threshold = distance_threshold
        self.metric_name = metric
        self.copy = copy

        # Aviso: Ward clássico é Euclidiano
        if metric != "euclidean_distance":
            warnings.warn(
                "WardMethod: o método de Ward clássico é definido para "
                "distância Euclidiana. Outras métricas são experimentais e "
                "não preservam a interpretação exata de variância mínima.",
                UserWarning,
            )

        # A função de distância concreta será configurada no fit()
        self.distance_func = None

        # Atributos de saída / estado
        self.labels_ = None
        self.sse_ = None
        self.sse_history = []           # guarda Δ de Ward a cada merge
        self.linkage_matrix_ = None
        self.merge_pairs_ = None
        self.n_leaves_ = None
        self.children_ = None

    # -------------------------------------------------------------------------
    def _setup_metric(self, X: np.ndarray):
        """
        Configura self.distance_func a partir de self.metric_name.
        Para Mahalanobis, usa covariância global e wrapper.
        """
        if self.metric_name == "mahalanobis_distance":
            cov = np.cov(X, rowvar=False)
            cov += 1e-6 * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov)

            def maha_wrapper(a, b, ic=inv_cov):
                return DistanceMetrics.mahalanobis(a, b, ic)

            self.distance_func = maha_wrapper
        else:
            self.distance_func = DistanceMetrics.get_metric(self.metric_name)

    # -------------------------------------------------------------------------
    def fit(self, X):
        """
        Constrói a árvore hierárquica de aglomeração com ligação Ward.
        """
        X = np.asarray(X, dtype=float)
        if self.copy:
            X = X.copy()

        n, d = X.shape
        self.n_leaves_ = n

        # Configura métrica agora que temos X (para Mahalanobis funcionar)
        self._setup_metric(X)

        # Cada ponto começa como um cluster
        max_nodes = 2 * n - 1
        sizes = np.zeros(max_nodes)
        centroids = np.zeros((max_nodes, d))

        sizes[:n] = 1
        centroids[:n] = X

        active = np.zeros(max_nodes, dtype=bool)
        active[:n] = True

        linkage = np.zeros((n - 1, 4))
        children = np.zeros((n - 1, 2), dtype=int)

        next_id = n
        iteration = 0

        def ward_delta(n_a, mu_a, n_b, mu_b):
            """
            Δ(i, j) = (n_a * n_b / (n_a + n_b)) * d(mu_a, mu_b)^2

            Para métrica Euclidiana, isso corresponde ao aumento de SSE
            (variância intra-cluster) ao unir os dois clusters.
            Com outras métricas, o valor é interpretado apenas como custo
            geométrico experimental.
            """
            dist = self.distance_func(mu_a, mu_b)
            return (n_a * n_b) / (n_a + n_b) * (dist ** 2)

        # Loop principal: unir clusters até sobrar 1
        while iteration < n - 1:
            valid = np.where(active[:next_id])[0]
            best_i, best_j = None, None
            best_delta = np.inf

            # Busca do par (i, j) com menor Δ de Ward
            for idx_a in range(len(valid)):
                a = valid[idx_a]
                for idx_b in range(idx_a + 1, len(valid)):
                    b = valid[idx_b]
                    cand = ward_delta(sizes[a], centroids[a],
                                      sizes[b], centroids[b])
                    if cand < best_delta:
                        best_delta, best_i, best_j = cand, a, b

            # Mesclar (best_i, best_j)
            ni, nj = sizes[best_i], sizes[best_j]
            mui, muj = centroids[best_i], centroids[best_j]

            new_size = ni + nj
            new_mean = (ni * mui + nj * muj) / new_size

            sizes[next_id] = new_size
            centroids[next_id] = new_mean

            # linkage: [id_cluster_1, id_cluster_2, ward_cost, tamanho_cluster]
            linkage[iteration] = [best_i, best_j, best_delta, new_size]
            children[iteration] = [best_i, best_j]
            self.sse_history.append(best_delta)

            # Atualiza ativos
            active[best_i] = False
            active[best_j] = False
            active[next_id] = True

            iteration += 1
            next_id += 1

        # Salvar resultados globais
        self.linkage_matrix_ = linkage
        self.merge_pairs_ = children
        self.children_ = children
        self.sse_ = float(np.sum(self.sse_history))

        # Se k ou distance_threshold foram definidos, produz labels
        if self.k is not None or self.distance_threshold is not None:
            self.labels_ = self._cut_tree(
                n_leaves=n,
                linkage=linkage,
                n_clusters=self.k,
                distance_threshold=self.distance_threshold,
            )

        return self

    # -------------------------------------------------------------------------
    def fit_predict(self, X):
        """
        Ajusta o modelo e retorna os rótulos finais.

        É necessário definir k ou distance_threshold no construtor
        para que os rótulos possam ser atribuídos.
        """
        self.fit(X)
        if self.labels_ is None:
            raise ValueError(
                "Defina k ou distance_threshold para usar fit_predict()."
            )
        return self.labels_

    # -------------------------------------------------------------------------
    def predict(self, X):
        """
        Ward é um método hierárquico aglomerativo: não suporta predição
        incremental consistente para novos pontos.
        """
        raise NotImplementedError(
            "O método Ward não realiza predição para novos pontos."
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _children_of(node_id, linkage, n_leaves):
        if node_id < n_leaves:
            return None
        idx = node_id - n_leaves
        return int(linkage[idx, 0]), int(linkage[idx, 1])

    @staticmethod
    def _leaves_under(node_id, linkage, n_leaves):
        """
        Retorna todos os índices de folhas (0..n_leaves-1) sob o nó node_id.
        """
        stack = [node_id]
        leaves = []
        while stack:
            node = stack.pop()
            children = WardMethod._children_of(node, linkage, n_leaves)
            if children is None:
                leaves.append(node)
            else:
                stack.extend(children)
        return np.array(leaves, dtype=int)

    @staticmethod
    def _cut_by_height(node_id, linkage, n_leaves, thr):
        """
        Corta a árvore em função de um limiar de "altura" (linkage[:, 2]),
        que aqui representa o custo incremental de Ward (ΔSSE).
        """
        children = WardMethod._children_of(node_id, linkage, n_leaves)
        if children is None:
            return [node_id]

        height = float(linkage[node_id - n_leaves, 2])
        if height <= thr:
            return [node_id]

        left, right = children
        return (
            WardMethod._cut_by_height(left, linkage, n_leaves, thr)
            + WardMethod._cut_by_height(right, linkage, n_leaves, thr)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _cut_tree(n_leaves, linkage, n_clusters=None, distance_threshold=None):
        """
        Constrói os rótulos finais com base em k ou em um limiar de custo (altura).
        """
        total_nodes = 2 * n_leaves - 1

        if n_clusters is not None:
            # Estratégia: começa da raiz e vai “cortando” os merges mais altos
            root = total_nodes - 1
            active = [root]

            while len(active) < n_clusters:
                # Considera apenas nós internos (>= n_leaves)
                internal_nodes = [node for node in active if node >= n_leaves]
                if not internal_nodes:
                    # Não há mais nós internos para dividir (todos são folhas)
                    break

                # Escolhe o nó com maior "altura" (maior custo em linkage[:, 2])
                node = max(
                    internal_nodes,
                    key=lambda nid: linkage[nid - n_leaves, 2]
                )
                active.remove(node)

                left, right = WardMethod._children_of(node, linkage, n_leaves)
                active.extend([left, right])

            labels = np.empty(n_leaves, dtype=int)
            for lab, node in enumerate(active):
                leaves = WardMethod._leaves_under(node, linkage, n_leaves)
                labels[leaves] = lab
            return labels

        if distance_threshold is not None:
            root = total_nodes - 1
            active_nodes = WardMethod._cut_by_height(
                root, linkage, n_leaves, distance_threshold
            )
            labels = np.empty(n_leaves, dtype=int)
            for lab, node in enumerate(active_nodes):
                leaves = WardMethod._leaves_under(node, linkage, n_leaves)
                labels[leaves] = lab
            return labels

        return None

    # -------------------------------------------------------------------------
    def plot_clusters(self, X, figsize=(7, 6)):
        """
        Plota clusters com cores distintas para dados 2D.
        """
        if self.labels_ is None:
            raise ValueError(
                "Execute fit com k ou distance_threshold definido antes de plotar."
            )

        X = np.asarray(X)
        if X.shape[1] != 2:
            raise ValueError(
                "plot_clusters espera dados 2D (duas features) para visualização."
            )

        plt.figure(figsize=figsize)
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=self.labels_,
            cmap="viridis",
            s=60,
            edgecolors="k",
        )
        plt.title(f"Ward Method (k={self.k}) - Métrica interna: {self.metric_name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------------------------
    def plot_sse(self):
        """
        Plota a evolução do Δ de Ward (incremento de SSE) a cada merge.
        """
        if len(self.sse_history) == 0:
            raise ValueError("Modelo não ajustado.")

        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(self.sse_history) + 1),
            self.sse_history,
            "o-",
        )
        plt.title("Evolução da variação intra-cluster (Δ de Ward)")
        plt.xlabel("Iteração (merge)")
        plt.ylabel("Δ de Ward (custo de fusão)")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------------------------
    def plot_dendrogram(self, figsize=(10, 5), truncate_mode=None, p=30):
        """
        Plota o dendrograma hierárquico com base na linkage_matrix_.
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Execute fit() antes de plotar o dendrograma.")

        plt.figure(figsize=figsize)
        dendrogram(
            self.linkage_matrix_,
            truncate_mode=truncate_mode,
            p=p,
            color_threshold=None,
        )
        plt.title(f"Dendrograma - Ward Method (métrica interna: {self.metric_name})")
        plt.xlabel("Amostras ou clusters")
        plt.ylabel("Δ de Ward (custo de fusão)")
        plt.grid(True)
        plt.show()
