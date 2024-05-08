import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import SnowballStemmer
from nltk.corpus import stopwords, words
from pandas import DataFrame
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, AffinityPropagation
from sklearn.decomposition import TruncatedSVD, PCA

stop_words = set(stopwords.words('english'))
english_vocab = set(word.lower() for word in words.words())
stem = SnowballStemmer('english')


def inspect(df: DataFrame):
    print(df.info())
    # RangeIndex: 2500 entries, 0 to 2499
    # Data columns (total 4 columns):
    #  #   Column       Non-Null Count  Dtype
    # ---  ------       --------------  -----
    #  0   Unnamed: 0   2500 non-null   int64
    #  1   headlines    2500 non-null   object
    #  2   description  2500 non-null   object
    #  3   content      2500 non-null   object
    # dtypes: int64(1), object(3)
    # memory usage: 78.3+ KB

    # Print the average amount of words in each cell of each column (so print 3 values)
    print(df.apply(lambda x: x.str.split().str.len().mean()))
    # headlines       13.9556
    # description     26.5372
    # content        228.3320

    # Print all words of length 2
    word_counts_2 = df.apply(lambda x: x.str.findall(r'\b\w{2}\b')).stack().explode().value_counts()
    print('Most common words length 2:\t' + ', '.join(word_counts_2.head(40).index))
    start_index = max(0, len(word_counts_2) // 3 - 20)
    end_index = min(len(word_counts_2), len(word_counts_2) // 3 + 20)
    print('Less common words length 2:\t' + ', '.join(word_counts_2.iloc[start_index:end_index].index))

    # Print all words of length 3
    word_counts_3 = df.apply(lambda x: x.str.findall(r'\b\w{3}\b')).stack().explode().value_counts()
    print('Most common words length 3:\t' + ', '.join(word_counts_3.head(40).index))
    start_index = max(0, len(word_counts_3) // 3 - 20)
    end_index = min(len(word_counts_3), len(word_counts_3) // 3 + 20)
    print('Less common words length 3:\t' + ', '.join(word_counts_3.iloc[start_index:end_index].index))

    # Print all words of length 4
    word_counts_4 = df.apply(lambda x: x.str.findall(r'\b\w{4}\b')).stack().explode().value_counts()
    print('Most common words length 4:\t' + ', '.join(word_counts_4.head(40).index))
    start_index = max(0, len(word_counts_4) // 3 - 20)
    end_index = min(len(word_counts_4), len(word_counts_4) // 3 + 20)
    print('Less common words length 4:\t' + ', '.join(word_counts_4.iloc[start_index:end_index].index))


def preprocess_column(column, min_count=0):
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


def explain(model, terms):
    if isinstance(model, KMeans):
        explain_kmeans(model, terms)
    elif isinstance(model, DBSCAN):
        explain_dbscan(model, terms)
    elif isinstance(model, AgglomerativeClustering):
        explain_agglomerative(model, terms)
    elif isinstance(model, Birch):
        explain_birch(model, terms)
    elif isinstance(model, AffinityPropagation):
        explain_dbscan(model, terms)


def explain_kmeans(kmeans_model, terms):
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    for i in range(len(order_centroids)):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print('--------------------------------')


def explain_dbscan(dbscan_model, terms):
    # Group the documents by cluster label
    clusters = {}
    for i, label in enumerate(dbscan_model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    for label, indices in clusters.items():
        print("Cluster %d:" % label)
        cluster_terms = []
        # Aggregate the terms from documents in the cluster
        for index in indices:
            cluster_terms.extend(terms[index].split())
        # Count the frequency of each term
        term_counts = {}
        for term in cluster_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        # Sort terms by frequency
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        # Print the top 10 terms
        for term, count in sorted_terms[:10]:
            print(' %s' % term)
        print('--------------------------------')


def explain_agglomerative(agglomerative_model, terms):
    # Get the cluster labels assigned by Agglomerative Clustering
    cluster_labels = agglomerative_model.labels_

    # Create a dictionary to store the terms associated with each cluster
    cluster_terms = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_terms:
            cluster_terms[label] = []
        cluster_terms[label].append(terms[i])

    # Print the top terms for each cluster
    for label, terms in cluster_terms.items():
        print(f"Cluster {label}:")
        # Print the top 10 terms for each cluster
        for term in terms[:10]:
            print(f" {term}")
        print('--------------------------------')


def explain_birch(birch_model, terms):
    # Get the cluster labels assigned by BIRCH
    cluster_labels = birch_model.labels_

    # Create a dictionary to store the terms associated with each cluster
    cluster_terms = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_terms:
            cluster_terms[label] = []
        cluster_terms[label].append(terms[i])

    # Print the top terms for each cluster
    for label, terms in cluster_terms.items():
        print(f"Cluster {label}:")
        # Print the top 10 terms for each cluster
        for term in terms[:10]:
            print(f" {term}")
        print('--------------------------------')


def explain_affinity_propagation(affinity_propagation_model, terms):
    # Get the cluster labels assigned by Affinity Propagation
    cluster_labels = affinity_propagation_model.labels_

    # Create a dictionary to store the terms associated with each cluster
    cluster_terms = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_terms:
            cluster_terms[label] = []
        cluster_terms[label].append(terms[i])

    # Print the top terms for each cluster
    for label, terms in cluster_terms.items():
        print(f"Cluster {label}:")
        # Print the top 10 terms for each cluster
        for term in terms[:10]:
            print(f" {term}")
        print('--------------------------------')


def plot_clusters(cluster_assignments, title='Number of documents in each cluster'):
    plt.hist(cluster_assignments, bins=len(set(cluster_assignments)))
    plt.xlabel('Cluster')
    plt.ylabel('Number of documents')
    plt.title(title)
    plt.show()


def plot_2d_datapoints(X, cluster_assignments=None):
    # Perform TruncatedSVD
    svd = TruncatedSVD(n_components=2)
    svd_components = svd.fit_transform(X)

    # Create a DataFrame for the SVD results
    svd_df = pd.DataFrame(data=svd_components, columns=['Component 1', 'Component 2'])

    # Plotting
    plt.figure(figsize=(10, 8))

    if cluster_assignments is not None:
        svd_df['Cluster'] = cluster_assignments
        num_clusters = len(set(cluster_assignments))
        for i in range(num_clusters):
            plt.scatter(svd_df.loc[svd_df['Cluster'] == i, 'Component 1'],
                        svd_df.loc[svd_df['Cluster'] == i, 'Component 2'],
                        label=f'Cluster {i}')
    else:
        plt.scatter(svd_df['Component 1'], svd_df['Component 2'], color='gray')

    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('2D SVD plot of document clusters')
    if cluster_assignments is not None:
        plt.legend()
    plt.show()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X.toarray())

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=pca_components, columns=['Component 1', 'Component 2'])

    # Plotting
    plt.figure(figsize=(10, 8))

    if cluster_assignments is not None:
        pca_df['Cluster'] = cluster_assignments
        num_clusters = len(set(cluster_assignments))
        for i in range(num_clusters):
            plt.scatter(pca_df.loc[pca_df['Cluster'] == i, 'Component 1'],
                        pca_df.loc[pca_df['Cluster'] == i, 'Component 2'],
                        label=f'Cluster {i}')
    else:
        plt.scatter(pca_df['Component 1'], pca_df['Component 2'], color='gray')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D PCA plot of document data points')
    if cluster_assignments is not None:
        plt.legend()
    plt.show()


def explain_clusters(labels, X_original, feature_names):
    num_clusters = np.max(labels) + 1

    # Calculate the mean TF-IDF score for each term in each cluster
    term_ratios = np.zeros((num_clusters, X_original.shape[1]))

    for i in range(num_clusters):
        # Find indices of documents in the cluster
        indices = np.where(labels == i)[0]
        # Aggregate TF-IDF scores by mean within the cluster
        term_ratios[i, :] = np.mean(X_original[indices], axis=0)

    # Print top terms for each cluster
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        top_terms = term_ratios[i].argsort()[::-1][:10]  # Get indices of top terms
        for ind in top_terms:
            print(f' {feature_names[ind]} ({term_ratios[i, ind]:.2f})', end=', ')
        print('\n--------------------------------')