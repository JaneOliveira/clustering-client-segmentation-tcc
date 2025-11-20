# clustering-client-segmentation-tcc

> Projeto de exploração e comparação de algoritmos de clusterização aplicados à segmentação de clientes, produtos, vendedores e pedidos, com base em múltiplos datasets reais e sintéticos. Foco em: qualidade dos agrupamentos, impacto da escolha da métrica de distância, interpretação de negócios e criação de um guia reprodutível para estudos de TCC.

## 1. Visão Geral
Este repositório implementa do zero (quando pertinente) e/ou adapta algoritmos clássicos de clusterização para análise de segmentação multieixo (clientes, produtos, vendedores e pedidos). Inclui um pipeline completo: pré-processamento, redução de dimensionalidade (PCA), aplicação de algoritmos, avaliação interna, comparação experimental entre métricas de distância e visualização estruturada.

## 2. Objetivos Principais
- Demonstrar diferenças estruturais entre algoritmos baseados em centróide (K-Means), representativos (K-Medoids), densidade (DBSCAN), hierarquia (Ward) e modelos probabilísticos (Expectation-Maximization / GMM).
- Investigar o impacto de métricas de distância (Euclidiana, Manhattan, Cosseno, Mahalanobis, Jensen–Shannon) na formação dos clusters e em métricas internas.
- Criar um framework simples para experimentos reprodutíveis em notebooks.
- Extrair insights de negócio (marketing, logística, satisfação, retenção) a partir dos agrupamentos.

## 3. Estrutura do Projeto
```
clustering-client-segmentation-tcc/
	data/DatasetRaw/            # Bases originais (Olist, Bank Marketing, Online Retail)
	notebooks/                  # Experimentos Jupyter (pré-processamento e análises)
	results/                    # Saídas e artefatos (ex: testes sintéticos)
	src/                        # Implementações de algoritmos e utilitários
		kmeans.py                 # K-Means manual com suporte a várias métricas
		kmedoids.py               # K-Medoids manual (busca local de medoides)
		dbscan.py                 # DBSCAN manual (densidade + métrica arbitrária)
		expectationmaximization.py# EM / GMM híbrido (probabilístico + modo experimental)
		wardmethod.py             # Ward hierárquico (linkage variância mínima)
		distance_metrics.py       # Coleção de métricas de distância
		evaluationmetrics.py      # Métricas internas / externas de avaliação
		func_auxiliar.py          # Funções auxiliares para testes e comparação
		plotsconfig.py            # Visualizações 2D / 3D e PCA
	README.md                   # Este documento
	requirements.txt            # Dependências mínimas
	LICENSE                     # Licença MIT
```

## 4. Algoritmos Implementados
| Algoritmo | Tipo | Características | Observações |
|-----------|------|-----------------|-------------|
| K-Means | Particional | Atualiza centróides pela média | SSE calculado sempre Euclidiano para comparação universal |
| K-Medoids | Particional | Representa cluster por ponto real (medoide) | Menos sensível a outliers que K-Means |
| DBSCAN | Densidade | Detecta ruído e clusters arbitrários | Métricas personalizadas afetam vizinhança (eps) |
| Expectation-Maximization (GMM) | Probabilístico | Estima parâmetros (μ, Σ, π) | Modo experimental quando métrica ≠ Mahalanobis |
| Ward (Hierárquico) | Hierárquico aglomerativo | Minimiza incremento de variância | Uso de métricas ≠ Euclidiana é experimental |

## 5. Métricas de Distância Suportadas (`distance_metrics.py`)
- `euclidean_distance`: norma L2 clássica
- `manhattan_distance`: soma das distâncias absolutas
- `cosine_distance`: 1 − cos(v1, v2)
- `mahalanobis_distance`: usa Σ⁻¹ global (regularizada)
- `jensen-shannon_distance`: distância (raiz da divergência) para vetores interpretados como distribuições

Uso dentro dos algoritmos: cada implementação configura internamente a função de distância. Para Mahalanobis, calcula-se a matriz de covariância global e aplica-se uma regularização leve (diagonal * 1e-6) para estabilidade numérica.

## 6. Métricas de Avaliação (`evaluationmetrics.py`)
| Métrica | Tipo | Interpretação |
|---------|------|---------------|
| Silhouette | Interna | Coesão vs separação ([-1,1]) — maior é melhor |
| Davies-Bouldin | Interna | Média da razão de dispersão vs separação — menor é melhor |
| Calinski-Harabasz | Interna | Razão variância inter / intra — maior é melhor |
| SSE Euclidiano | Interna | Soma dos erros quadráticos usando distância Euclidiana — menor é melhor |
| Homogeneity | Externa (quando y_true) | Cada cluster contém somente uma classe |
| Completeness | Externa | Cada classe está contida em um único cluster |
| Fowlkes-Mallows | Externa | Harmônica baseada em precisão e recall de pares |

Importante: Mesmo quando um algoritmo utiliza outra métrica para formar clusters (ex.: Mahalanobis no K-Means ou DBSCAN com Cosseno), o SSE reportado é sempre calculado com distância Euclidiana. Isso padroniza a comparação geométrica entre métodos.

## 7. Datasets Disponíveis (`data/DatasetRaw/`)
### Olist (E-commerce Brasileiro)
Arquivos: clientes, geolocalização, pedidos, itens, pagamentos, reviews, produtos, sellers e tradução de categoria.
Principais eixos analíticos: fidelidade, ticket médio, logística (frete, atraso), satisfação (review_score), categoria de produto, dispersão geográfica.

### Bank Marketing
Base de campanhas de marketing direto (telemarketing). Usável para segmentar perfis de clientes por resposta, renda, profissão, etc.

### Online Retail
Transações de e-commerce (inclui sazonalidade, países, códigos de produto). Ideal para comparar segmentação por produto vs por cliente.

### Sintético (Blobs)
Gerado via `generate_synthetic_blobs` para validação rápida de implementação e comparação entre métricas.

## 8. Pipeline Recomendado
1. Pré-processamento: limpeza (remover registros incompletos / não entregues), encoding categórico (one-hot / IDs), normalização de variáveis quantitativas (exceto coordenadas se utilizadas geograficamente).
2. PCA (opcional): reduzir dimensionalidade para visualização (2D/3D) mantendo >= ~80% de variância explicada.
3. Execução de algoritmos: varrer intervalos (K para K-Means / K-Medoids), estimar `eps` (k-distance) para DBSCAN, escolher número de componentes (BIC/AIC) para EM.
4. Avaliação: Silhouette, CH, DB, SSE Euclidiano padronizado.
5. Interpretação: agregações por cluster (médias, distribuição de categorias, ticket, atraso, satisfação, geografia).
6. Visualização: `plotsconfig.py` (2D/3D), dendrograma (Ward), elipses Gaussianas (EM probabilístico), mapa geográfico (extensível com `folium`).

## 9. Instalação
Requer Python >= 3.11 (recomendado). Dependências mínimas:
```
pip install -r requirements.txt
```
Criação opcional de ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 10. Uso Rápido (Exemplo Sintético)
```python
from func_auxiliar import generate_synthetic_blobs
from kmeans import KMeans
from evaluationmetrics import EvaluationMetrics

# 1. Gerar dados
X, _ = generate_synthetic_blobs(n_samples=800, centers=4, cluster_std=0.6, random_state=42)

# 2. Rodar K-Means com métrica Euclidiana
model = KMeans(k=4, metric_name="euclidean_distance", random_state=42)
centroids, labels = model.fit(X)

# 3. Avaliar
evaluator = EvaluationMetrics(X, labels)
print(evaluator.summary())  # Silhouette, DB, CH
print("SSE:", evaluator.sse_euclidean())

# 4. Visualizar
model.plot_clusters(X, labels)
model.plot_sse()
```

### Comparação entre Métricas (Exemplo Resumido)
```python
from func_auxiliar import generate_synthetic_blobs, calculate_clustering_metrics
from kmeans import KMeans

X, _ = generate_synthetic_blobs()
metricas = ["euclidean_distance", "manhattan_distance", "cosine_distance", "mahalanobis_distance"]
resultados = []
for m in metricas:
		km = KMeans(k=3, metric_name=m, random_state=0)
		centroids, labels = km.fit(X)
		r = calculate_clustering_metrics("kmeans", m, runtime=0.0, X=X, labels=labels)
		resultados.append(r)

for r in resultados:
		print(r["internal_metric"], "Silhouette:", round(r["silhouette"], 3))
```

## 11. Notebooks
Local: `notebooks/`
- Contêm experimentos específicos por dataset (pré-processamento, normalização, PCA, testes de algoritmos).
- Exemplos: `ExperimentoDatasetOlist.ipynb`, `ExperimentoDatasetRetailOnline.ipynb`.
- Recomenda-se executar em ordem (pré-processamento → clustering → avaliação → visualização → interpretação).

## 12. Resultados
Pasta `results/` armazena saídas de testes (ex.: dataset sintético), possibilitando rastrear parâmetros usados e métricas obtidas.

## 13. Oportunidades Analíticas (Resumo Estruturado)
### Clientes (customer_unique_id)
Perguntas: frequência vs recência, ticket médio vs número de parcelas, padrões geográficos, relação atraso-entrega vs satisfação, diversidade de categorias e risco de churn.
Insight exemplo: “Cluster C2: Sudeste, 2 parcelas, casa & decoração, ticket médio ~R$350, review_score > 4.5.”

### Produtos (product_id)
Perguntas: preço × volume, custo logístico, sazonalidade, devoluções, margem potencial.
Insight exemplo: “Cluster P1: leves, baratos, alta rotatividade (promoções); Cluster P3: pesados, caros, frete alto e mais atrasos.”

### Vendedores (seller_id)
Perguntas: volume × atraso, avaliação média, concentração regional, diversidade de categorias, risco de churn de vendedores.
Insight exemplo: “Cluster S0: microvendedores Nordeste, baixo volume, alto atraso; Cluster S2: grandes vendedores Sudeste, entregas rápidas, ticket alto.”

### Pedidos (order_id)
Perguntas: tempo de entrega, ticket total regional, sazonalidade (picos), frete elevado concentrado.

### Interclusters
Combinações: Cliente+Produto, Cliente+Região, Produto+Vendedor, Cliente+Pagamento, Cliente+Entrega. Geram narrativas cruzadas para priorização de campanhas e logística.

### Ideias de Histórias para TCC
- Segmentação de clientes (RFM estendido + satisfação + geografia)
- Clusterização de produtos para otimização logística
- Regiões de alto potencial de venda (geolocalização + categorias)
- Correlação comportamento de compra vs satisfação
- Perfis de vendedores vs impacto em avaliações e ticket

## 14. Plano de Ação (Detalhado) – Base Olist
Resumo dos passos já codificados nos notebooks: pré-processamento → PCA → seleção de K / eps / componentes → execução → avaliação (Silhouette, CH, DB, SSE) → visualização (2D/3D, dendrograma, mapas) → interpretação e relato.
Notas:
- SSE Euclidiano padronizado torna comparável K-Means, K-Medoids, DBSCAN, Ward e EM.
- Modo EM com métrica ≠ Mahalanobis fornece “pseudo-soft-clustering” (não é probabilidade formal).
- Ward com métricas não Euclidianas é experimental (perde interpretação clássica de variância mínima).

## 15. Trabalhos Futuros
- Implementar avaliação externa cruzando rótulos pseudo-supervisionados (quando disponível).
- Adicionar HDBSCAN para densidade adaptativa.
- Incluir métricas de estabilidade (ex.: variação do Silhouette em bootstrap).
- Exportação de relatórios automatizados (Markdown / HTML) por experimento.
- Integração com Folium/KeplerGL para mapas avançados.

## 16. Licença
Licenciado sob MIT – ver arquivo `LICENSE`.

## 17. Referências Rápidas
- DBSCAN original: Ester et al. (1996)
- Ward: Ward (1963) – variância mínima
- GMM / EM: Dempster, Laird & Rubin (1977)
- Métricas internas: Kaufman & Rousseeuw (Silhouette), Davies & Bouldin, Calinski & Harabasz

---
Caso deseje expandir ou gerar novos experimentos, consulte os notebooks e adapte os parâmetros de métricas conforme a natureza dos dados (denso vs esparso, distribuição vetorial, etc.). Contribuições são bem-vindas.

Obrigada por ler ate aqui ❤️