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
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')

from utils import inspect, explain

# Load dataset
df = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])
original_df = df.copy()

stop_words = set(stopwords.words('english'))
english_vocab = set(word.lower() for word in words.words())
stem = SnowballStemmer('english')

# Inspect the dataset
# inspect(df)

# Based on the steps above, create a function called preprocess_column that preprocesses a column
def preprocess_column(column, min_count = 0):
    # Lowercase
    column = column.apply(lambda x: x.lower().strip())
    # Remove special characters
    column = column.replace(r'[^a-zA-Z]', r' ', regex=True)
    # Remove words of length 1 or 2
    column = column.replace(r'\b\w{1,2}\b', r'', regex=True)
    # Remove multiple spaces
    column = column.replace(r'\s+', r' ', regex=True)
    # Remove leading and trailing spaces
    column = column.apply(lambda x: x.strip())
    # Remove stopwords
    column = column.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    # Remove non-english words
    # column = column.apply(lambda x: " ".join(x for x in str(x).split() if x in english_vocab or len(x) > 3))
    # Remove words whose length is more than 3 and have no syllable
    # Print count of words that have length more than 3 and have no syllable
    column = column.apply(lambda x: " ".join(x for x in str(x).split() if len(x) <= 3 or any(v in x for v in 'aeiou')))
    # Stemming
    # column = column.apply(lambda x: ' '.join([stem.stem(word) for word in str(x).split()]))
    # Create dictionary of words and their frequency
    word_freq = pd.Series(' '.join(column).split()).value_counts()
    # Remove all elements that only occur once
    column = column.apply(lambda x: ' '.join([word for word in x.split() if word_freq[word] > min_count]))
    return column


# Preprocess the content column
df['headlines'] = preprocess_column(df['headlines'], min_count=2)
df['description'] = preprocess_column(df['description'], min_count=2)
df['content'] = preprocess_column(df['content'], min_count=10)

df.to_excel('assignment3_articles_preprocessed.xlsx', index=False)


# Define a custom scoring function
def custom_score(estimator, X):
    estimator.fit(X)
    cluster_assignments = estimator.labels_
    hist, _ = np.histogram(cluster_assignments, bins=num_clusters)
    std = np.std(hist)
    return -std

def silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_assignments = estimator.labels_
    return silhouette_score(X, cluster_assignments)

def custom_score_toarray(estimator, X):
    estimator.fit(X)
    cluster_assignments = estimator.labels_
    hist, _ = np.histogram(cluster_assignments, bins=num_clusters)
    std = np.std(hist)
    return -std


parameters = {
    # "AffinityPropagation": {"damping": [0.5, 0.7, 0.9]},
    # "Birch": {"n_clusters": [2]},
    # "KMeans": {"n_clusters": list(range(3, 10)), "init": ["k-means++", "random"], "n_init": [10, 20, 30], "max_iter": [50, 100, 200], "tol": [1e-4, 1e-2]},
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

# Iterate over each clustering model and perform grid search
# for model_name in parameters.keys():
#     print(f"Performing grid search for {model_name}")
#     start = time.time()
#     model = globals()[model_name]()
#     param_grid = parameters[model_name]
#     grid_search = GridSearchCV(model, param_grid, scoring=custom_score, error_score='raise', verbose=0, n_jobs=-1, cv=3)
#     grid_search.fit(X)
#     print(f"Best parameters for {model_name}: {grid_search.best_params_}")
#     # Run again with the best parameters
#     model = globals()[model_name](**grid_search.best_params_)
#     model.fit(X)
#     cluster_assignments = model.labels_
#     # Create a bar chart showing the number of documents in each cluster
#     plt.hist(cluster_assignments, bins=num_clusters)
#     plt.xlabel('Cluster')
#     plt.ylabel('Number of documents')
#     plt.title(f'Documents per cluster for {model_name}')
#     plt.show()
#     # Chronometer
#     end = time.time()
#     print(f"Time taken: {end - start}")

# Create bag of words of all three columns, give 'headlines' a weight of 3, 'description' a weight of 2 and 'content' a weight of 1
# Assuming df is your DataFrame
vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()

# Transforming each column separately
# combined_text = df['headlines'] + ' ' + df['description']
combined_text = df['headlines'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description'] + ' ' + df['content']
# combined_text = df['headlines'] + ' ' + df['headlines'] + ' ' + df['headlines'] + ' ' + df['description'] + ' ' + df['description']
X = vectorizer.fit_transform(combined_text)
print(X.nnz)
print(X.shape)


# for model_name in parameters_toarray.keys():
#     print(f"Performing grid search for {model_name}")
#     model = globals()[model_name]()
#     param_grid = parameters_toarray[model_name]
#     grid_search = GridSearchCV(model, param_grid, scoring=custom_score_toarray, error_score='raise')
#     grid_search.fit(X.toarray())
#     print(f"Best parameters for {model_name}: {grid_search.best_params_}")





# Assuming X is your bag-of-words matrix
num_clusters = 8
# Initialize KMeans clustering with 10 clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# Use other clustering algorithnm
# kmeans = DBSCAN(eps=0.6, min_samples=15, metric='cosine')
# Another clustering algorithm
# kmeans = AgglomerativeClustering(n_clusters=num_clusters)
# kmeans = Birch(n_clusters=num_clusters)

# Fit the KMeans model to your data
# kmeans.fit(X)
kmeans.fit(X.toarray())
# Get the cluster assignments for each document
cluster_assignments = kmeans.labels_
# Print the cluster assignments for each document
print(cluster_assignments)
# Add this to the original dataframe and output it to a new Excel file
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
explain(kmeans, vectorizer.get_feature_names_out())

