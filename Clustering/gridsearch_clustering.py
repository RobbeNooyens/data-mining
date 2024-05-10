import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
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

from Clustering.utils import plot_documents_per_cluster, preprocess_column






# Load dataset
df = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])

# Preprocess the content column
df['headlines'] = preprocess_column(df['headlines'], min_count=2)
df['description'] = preprocess_column(df['description'], min_count=2)
df['content'] = preprocess_column(df['content'], min_count=10)

vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()

# combined_text = df['headlines'] + ' ' + df['description']
combined_text = df['headlines'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description']
X = vectorizer.fit_transform(combined_text)

# Iterate over each clustering model and perform grid search
for model_name in parameters.keys():
    print(f"Running grid search for {model_name}")
    start = time.time()
    # ================================================
    model = globals()[model_name]()
    param_grid = parameters[model_name]
    best_params = tune_model(model, X, param_grid)
    # Run again with the best parameters
    model = globals()[model_name](**best_params)
    model.fit(X)
    cluster_assignments = model.labels_
    plot_documents_per_cluster(cluster_assignments, model_name)
    explain(model, vectorizer.get_feature_names_out())
    # ================================================
    end = time.time()
    print(f"Time taken: {end - start}")
