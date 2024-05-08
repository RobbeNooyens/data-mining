import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    KMeans,
    BisectingKMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    SpectralBiclustering,
    SpectralCoclustering,
    cluster_optics_dbscan,
    cluster_optics_xi,
    affinity_propagation,
    dbscan,
    estimate_bandwidth,
    get_bin_seeds,
    k_means,
    kmeans_plusplus,
    linkage_tree,
    mean_shift,
    spectral_clustering,
    ward_tree,
    HDBSCAN,
)
from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords, words
import nltk
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.model_selection import GridSearchCV
from umap import UMAP

nltk.download('stopwords')

from utils import inspect, explain, preprocess_column, plot_2d_datapoints, explain_clusters

# Load dataset
df = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])
original_df = df.copy()

# Inspect the dataset
# inspect(df)

# Preprocess the content column
df['headlines'] = preprocess_column(df['headlines'], min_count=2)
df['description'] = preprocess_column(df['description'], min_count=2)
df['content'] = preprocess_column(df['content'], min_count=10)

df.to_excel('assignment3_articles_preprocessed.xlsx', index=False)

vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()

# Transforming each column separately
# combined_text = df['headlines'] + ' ' + df['description']
combined_text = df['headlines'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description']
X = vectorizer.fit_transform(combined_text)
X_original = X.copy()

umap = UMAP(n_components=2)
X = umap.fit_transform(X.toarray())


num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans = DBSCAN(eps=0.6, min_samples=15, metric='cosine')
# kmeans = AgglomerativeClustering(n_clusters=num_clusters)
# kmeans = Birch(n_clusters=num_clusters)

kmeans.fit(X)
# kmeans.fit(X.toarray())
cluster_assignments = kmeans.labels_
print(cluster_assignments)
original_df['cluster'] = cluster_assignments
# Add a column with 'headlines_word_count', 'description_word_count' and 'content_word_count' to the original_df
original_df['headlines_word_count'] = original_df['headlines'].apply(lambda x: len(x.split()))
original_df['description_word_count'] = original_df['description'].apply(lambda x: len(x.split()))
original_df['content_word_count'] = original_df['content'].apply(lambda x: len(x.split()))
original_df.to_excel('assignment3_articles_clustered.xlsx', index=False)

# Create a bar chart showing the number of documents in each cluster
plt.hist(cluster_assignments, bins=num_clusters)
plt.xlabel('Cluster')
plt.ylabel('Number of documents')
plt.title('Number of documents in each cluster')
plt.show()
# explain(kmeans, vectorizer.get_feature_names_out())
# plot_2d_datapoints(X, cluster_assignments)


# tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
# tsne_results = tsne.fit_transform(X.toarray())  # Make sure X is dense if it's not already
#
# # Plotting
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_assignments, cmap='viridis')
# plt.colorbar(label='Cluster label')
# plt.title('2D t-SNE plot of document data points')
# plt.show()

# umap = UMAP(n_components=2)
# umap_results = umap.fit_transform(X.toarray())

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis')
plt.colorbar(label='Cluster label')
plt.title('2D UMAP plot of document data points')
plt.show()

explain_clusters(kmeans.labels_, X_original, vectorizer.get_feature_names_out())