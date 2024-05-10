import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, SpectralClustering, Birch, \
    AffinityPropagation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

from utils import preprocess_column, print_cluster_explanation, inspect, plot_cluster_assignments, \
    plot_documents_per_cluster, evaluate_clustering, print_cluster_explanation_table, bag_of_words, tune_model

# ================  Load Dataset  ================
df_original = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])

# ================  Inspect Dataset  ================
inspect(df_original)

# ================  Preprocessing  ================
df = df_original.copy()
df['headlines'] = preprocess_column(df['headlines'], min_count=2)
df['description'] = preprocess_column(df['description'], min_count=2)
df['content'] = preprocess_column(df['content'], min_count=10)
df.to_excel('assignment3_articles_preprocessed.xlsx', index=False)

df_stemmed = df_original.copy()
df_stemmed['headlines'] = preprocess_column(df_stemmed['headlines'], min_count=2, apply_stemming=True)
df_stemmed['description'] = preprocess_column(df_stemmed['description'], min_count=2, apply_stemming=True)
df_stemmed['content'] = preprocess_column(df_stemmed['content'], min_count=10, apply_stemming=True)

# ================  Bag-of-words using TF-IDF  ================
X, feature_names = bag_of_words(df)
X_stemmed, feature_names_stemmed = bag_of_words(df_stemmed)
X_original = X.copy()

# ================  Dimensionality Reduction  ================
X_UMAP = UMAP(n_components=2).fit_transform(X.toarray())
X_PCA = PCA(n_components=2).fit_transform(X.toarray())
X_TSNE = TSNE(n_components=2).fit_transform(X.toarray())
X_KernelPCA = KernelPCA(n_components=2).fit_transform(X.toarray())
X_MDS = MDS(n_components=2).fit_transform(X.toarray())
X_IsoMap = Isomap(n_components=2).fit_transform(X.toarray())
X_LocallyLinearEmbedding = LocallyLinearEmbedding(n_components=2).fit_transform(X.toarray())
X_SpectralEmbedding = SpectralEmbedding(n_components=2).fit_transform(X.toarray())
X_UMAP_stemmed = UMAP(n_components=2).fit_transform(X_stemmed.toarray())

# ================  Visualize Clusters  ================
plot_cluster_assignments(X_UMAP, "UMAP")
plot_cluster_assignments(X_PCA, "PCA")
plot_cluster_assignments(X_TSNE, "TSNE")
plot_cluster_assignments(X_KernelPCA, "KernelPCA")
plot_cluster_assignments(X_MDS, "MDS")
plot_cluster_assignments(X_IsoMap, "IsoMap")
plot_cluster_assignments(X_LocallyLinearEmbedding, "LocallyLinearEmbedding")
plot_cluster_assignments(X_SpectralEmbedding, "SpectralEmbedding")
plot_cluster_assignments(X_UMAP_stemmed, "UMAP Stemmed")

# ================  KMeans Clustering  ================
kmeans_UMAP = KMeans(n_clusters=5, random_state=0)
kmeans_TSNE = KMeans(n_clusters=5, random_state=0)
kmeans_UMAP_stemmed = KMeans(n_clusters=5, random_state=0)

kmeans_UMAP.fit(X_UMAP)
kmeans_TSNE.fit(X_TSNE)
kmeans_UMAP_stemmed.fit(X_stemmed)

plot_documents_per_cluster(kmeans_UMAP_clusters := kmeans_UMAP.labels_, "Document per cluster - UMAP KMeans(5)")
plot_documents_per_cluster(kmeans_TSNE_clusters := kmeans_TSNE.labels_, "Document per cluster - TSNE KMeans(5)")
plot_documents_per_cluster(kmeans_UMAP_stemmed_clusters := kmeans_UMAP_stemmed.labels_, "Document per cluster - UMAP Stemmed KMeans(5)")

plot_cluster_assignments(X_UMAP, "UMAP KMeans(5)", kmeans_UMAP_clusters)
plot_cluster_assignments(X_TSNE, "TSNE KMeans(5)", kmeans_TSNE_clusters)
plot_cluster_assignments(X_UMAP_stemmed, "UMAP Stemmed KMeans(5)", kmeans_UMAP_stemmed_clusters)

print_cluster_explanation_table(kmeans_UMAP.labels_, X_original, feature_names)
print_cluster_explanation_table(kmeans_TSNE.labels_, X_original, feature_names)
print_cluster_explanation_table(kmeans_UMAP_stemmed.labels_, X_original, feature_names)

evaluate_clustering(X_UMAP, kmeans_UMAP.labels_, "UMAP")
evaluate_clustering(X_TSNE, kmeans_TSNE.labels_, "TSNE")
evaluate_clustering(X_UMAP_stemmed, kmeans_UMAP_stemmed.labels_, "UMAP Stemmed")

df_original['UMAP KMeans(5)'] = kmeans_UMAP_clusters
df_original['TSNE KMeans(5)'] = kmeans_TSNE_clusters
df_original.to_excel('assignment3_articles_clustered.xlsx', index=False)

# ================  Extended Clustering  ================
dbscan_UMAP = DBSCAN()
hdbscan_UMAP = HDBSCAN(max_cluster_size=10)
agglo_UMAP = AgglomerativeClustering(n_clusters=5)
spectral_UMAP = SpectralClustering(n_clusters=5)
birch_UMAP = Birch(n_clusters=5)
affinity_UMAP = AffinityPropagation()

dbscan_UMAP.fit(X_UMAP)
hdbscan_UMAP.fit(X_UMAP)
agglo_UMAP.fit(X_UMAP)
spectral_UMAP.fit(X_UMAP)
birch_UMAP.fit(X_UMAP)
affinity_UMAP.fit(X_UMAP)

plot_documents_per_cluster(dbscan_UMAP_clusters := dbscan_UMAP.labels_, "Document per cluster - UMAP DBSCAN")
plot_documents_per_cluster(hdbscan_UMAP_clusters := hdbscan_UMAP.labels_, "Document per cluster - UMAP HDBSCAN")
plot_documents_per_cluster(agglo_UMAP_clusters := agglo_UMAP.labels_, "Document per cluster - UMAP Agglomerative")
plot_documents_per_cluster(spectral_UMAP_clusters := spectral_UMAP.labels_, "Document per cluster - UMAP Spectral")
plot_documents_per_cluster(birch_UMAP_clusters := birch_UMAP.labels_, "Document per cluster - UMAP Birch")
plot_documents_per_cluster(affinity_UMAP_clusters := affinity_UMAP.labels_, "Document per cluster - UMAP Affinity")

plot_cluster_assignments(X_UMAP, "UMAP DBSCAN", dbscan_UMAP_clusters)
plot_cluster_assignments(X_UMAP, "UMAP HDBSCAN", hdbscan_UMAP_clusters)
plot_cluster_assignments(X_UMAP, "UMAP Agglomerative", agglo_UMAP_clusters)
plot_cluster_assignments(X_UMAP, "UMAP Spectral", spectral_UMAP_clusters)
plot_cluster_assignments(X_UMAP, "UMAP Birch", birch_UMAP_clusters)
plot_cluster_assignments(X_UMAP, "UMAP Affinity", affinity_UMAP_clusters)

evaluate_clustering(X_UMAP, dbscan_UMAP.labels_, "UMAP DBSCAN")
evaluate_clustering(X_UMAP, hdbscan_UMAP.labels_, "UMAP HDBSCAN")
evaluate_clustering(X_UMAP, agglo_UMAP.labels_, "UMAP Agglomerative")
evaluate_clustering(X_UMAP, spectral_UMAP.labels_, "UMAP Spectral")
evaluate_clustering(X_UMAP, birch_UMAP.labels_, "UMAP Birch")
evaluate_clustering(X_UMAP, affinity_UMAP.labels_, "UMAP Affinity")


# ================  Optimization using GridSearch  ================
parameters = {
    "KMeans": {
        "n_clusters": [4, 5, 6, 7, 8],
        "init": ["k-means++", "random"],
        "n_init": [10, 20, 30],
        "max_iter": [50, 100, 200],
        "tol": [1e-4, 1e-2],
    },
    "AgglomerativeClustering": {
        "n_clusters": [4, 5, 6],
        "linkage": ["complete", "average", "single"],
        "metric": ["euclidean", "l1", "l2", "manhattan", "cosine"]
    },
    "SpectralClustering": {
        "n_clusters": [4, 5, 6],
        "n_neighbors": [10, 20, 30],
    },
    "Birch": {
        "n_clusters": [4, 5, 6],
        "threshold": [0.5, 0.75, 1],
    },
    "AffinityPropagation": {
        "damping": [0.5, 0.75, 0.9],
        "max_iter": [200, 300, 400],
    },
    "DBSCAN": {
        "eps": [0.5, 0.75, 1],
        "min_samples": [5, 10, 15],
        "metric": ["euclidean", "manhattan"]
    },
    "HDBSCAN": {
        "min_cluster_size": [5, 10, 15],
        "min_samples": [5, 10, 15],
        "cluster_selection_epsilon": [0.5, 0.75, 1],
    }
}

for model_name in parameters.keys():
    model = globals()[model_name]()
    param_grid = parameters[model_name]
    best_params = tune_model(model, X_UMAP, param_grid)
    # Run again with the best parameters
    model = globals()[model_name](**best_params)
    model.fit(X_UMAP)
    cluster_assignments = model.labels_
    plot_documents_per_cluster(cluster_assignments, f"Document per cluster - UMAP {model_name} GridSearch")
    plot_cluster_assignments(X_UMAP, f"UMAP {model_name} GridSearch", cluster_assignments)
    evaluate_clustering(X_UMAP, cluster_assignments, f"UMAP {model_name} GridSearch")
