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

from Clustering.utils import plot_documents_per_cluster, preprocess_column, explain


def tune_model(model, X, parameters):
    grid_search = GridSearchCV(model, parameters, scoring=silhouette_scorer, error_score='raise', verbose=0, n_jobs=-1, cv=3)
    grid_search.fit(X)
    print(f"Best parameters for {model}: {grid_search.best_params_}")
    return grid_search.best_params_


# Define a custom scoring function
def std_score(estimator, X):
    estimator.fit(X)
    hist, _ = np.histogram(estimator.labels_, bins=len(np.unique(estimator.labels_)))
    return -np.std(hist)


def silhouette_scorer(estimator, X):
    estimator.fit(X)
    return silhouette_score(X, estimator.labels_)

AffinityPropagation()
parameters = {
    # "AffinityPropagation": {"damping": [0.5, 0.7, 0.9], "max_iter": [200, 400, 600], "convergence_iter": [15, 20, 25]},
    "Birch": {"n_clusters": [3,6,9]},
    # "KMeans": {"n_clusters": list(range(3, 10)), "init": ["k-means++", "random"], "n_init": [10, 20, 30],
    #            "max_iter": [50, 100, 200], "tol": [1e-4, 1e-2]},
    # "BisectingKMeans": {"n_clusters": [2, 3, 4, 5], "n_init": [10, 20, 30], "max_iter": [50, 100, 200]},
    # "MiniBatchKMeans": {"n_clusters": [2, 3, 4, 5], "init": ["k-means++", "random"], "n_init": [10, 20, 30], "max_iter": [50, 100, 200]},
    # "HDBSCAN": {"min_cluster_size": [10, 20, 30], "min_samples": [5, 10, 15], "cluster_selection_epsilon": [0.5, 1.0, 1.5]},
}

parameters_toarray = {
    # "AgglomerativeClustering": {"n_clusters": [2, 3, 4, 5]},
    "DBSCAN": {"eps": [0.5, 1.0, 1.5], "min_samples": [5, 10, 15], "metric": ['euclidean', 'cosine']},
    # "OPTICS": {"min_samples": [5, 10, 15]},
    # "MeanShift": {"bandwidth": [0.1, 0.5, 1.0]},
}

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
