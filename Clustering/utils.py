from pandas import DataFrame
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch


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

def explain(model, terms):
    if isinstance(model, KMeans):
        explain_kmeans(model, terms)
    elif isinstance(model, DBSCAN):
        explain_dbscan(model, terms)
    elif isinstance(model, AgglomerativeClustering):
        explain_agglomerative(model, terms)
    elif isinstance(model, Birch):
        explain_birch(model, terms)

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