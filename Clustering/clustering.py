import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

from utils import preprocess_column, print_cluster_explanation, inspect, plot_cluster_assignments, \
    plot_documents_per_cluster, evaluate_clustering, print_cluster_explanation_table

# Load dataset
df = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])
df_original = df.copy()

# Inspect the dataset
inspect(df)

# Preprocess the content column
df['headlines'] = preprocess_column(df['headlines'], min_count=2)
df['description'] = preprocess_column(df['description'], min_count=2)
df['content'] = preprocess_column(df['content'], min_count=10)
df.to_excel('assignment3_articles_preprocessed.xlsx', index=False)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['headlines'] + ' ' + df['description'] + ' ' + df['content'])
X_original = X.copy()
print(X.shape, X.nnz)

X_UMAP = UMAP(n_components=2).fit_transform(X.toarray())
plot_cluster_assignments(X_UMAP, "UMAP")
# X_PCA = PCA(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_PCA, "PCA")
X_TSNE = TSNE(n_components=2).fit_transform(X.toarray())
plot_cluster_assignments(X_TSNE, "TSNE")
# X_KernelPCA = KernelPCA(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_KernelPCA, "KernelPCA")
# X_MDS = MDS(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_MDS, "MDS")
# X_IsoMap = Isomap(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_IsoMap, "IsoMap")
# X_LocallyLinearEmbedding = LocallyLinearEmbedding(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_LocallyLinearEmbedding, "LocallyLinearEmbedding")
# X_SpectralEmbedding = SpectralEmbedding(n_components=2).fit_transform(X.toarray())
# scatter_plot(X_SpectralEmbedding, "SpectralEmbedding")

# Fit basic KMeans clustering
kmeans_UMAP = KMeans(n_clusters=5, random_state=0)
kmeans_TSNE = KMeans(n_clusters=5, random_state=0)
# kmeans = DBSCAN()
# kmeans = AgglomerativeClustering(n_clusters=num_clusters)
# kmeans = Birch(n_clusters=num_clusters)

kmeans_UMAP.fit(X_UMAP)
kmeans_TSNE.fit(X_TSNE)

plot_documents_per_cluster(kmeans_UMAP_clusters := kmeans_UMAP.labels_, "Document per cluster - UMAP KMeans(5)")
plot_documents_per_cluster(kmeans_TSNE_clusters := kmeans_TSNE.labels_, "Document per cluster - TSNE KMeans(5)")

plot_cluster_assignments(X_UMAP, "UMAP KMeans(5)", kmeans_UMAP_clusters)
plot_cluster_assignments(X_TSNE, "TSNE KMeans(5)", kmeans_TSNE_clusters)

print_cluster_explanation_table(kmeans_UMAP.labels_, X_original, vectorizer.get_feature_names_out())
print_cluster_explanation_table(kmeans_TSNE.labels_, X_original, vectorizer.get_feature_names_out())

evaluate_clustering(X_UMAP, kmeans_UMAP.labels_, "UMAP")
evaluate_clustering(X_TSNE, kmeans_TSNE.labels_, "TSNE")


df_original['UMAP KMeans(5)'] = kmeans_UMAP_clusters
df_original['TSNE KMeans(5)'] = kmeans_TSNE_clusters
df_original.to_excel('assignment3_articles_clustered.xlsx', index=False)
