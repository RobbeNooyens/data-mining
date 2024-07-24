import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import SnowballStemmer
from nltk.corpus import stopwords, words
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

# Download words
nltk.download('stopwords')
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
    # Most common words length 2:	to, in, of, on, is, at, as, it, by, an, be, he, Rs, up, or, In, It, we, He, 10, AI, my, We, 12, if, so, do, no, 15, 19, me, 20, As, 30, US, UG, On, 11, 50, 17
    # Less common words length 2:	UP, 04, CS, ED, FE, MP, M3, Or, ho, NZ, Le, Mr, 02, ki, Ek, MA, 06, ko, ka, bi, Ka, 08, OS, St, El, QR, se, IP, SS, GB, FY, ID, Q3, Me, 4G, SC, au, Na, D1, hi

    # Print all words of length 3
    word_counts_3 = df.apply(lambda x: x.str.findall(r'\b\w{3}\b')).stack().explode().value_counts()
    print('Most common words length 3:\t' + ', '.join(word_counts_3.head(40).index))
    start_index = max(0, len(word_counts_3) // 3 - 20)
    end_index = min(len(word_counts_3), len(word_counts_3) // 3 + 20)
    print('Less common words length 3:\t' + ', '.join(word_counts_3.iloc[start_index:end_index].index))
    # Most common words length 3:	the, and, for, The, was, has, his, are, per, not, its, who, you, had, but, can, all, out, one, new, now, her, You, two, get, top, him, day, Now, how, she, Cup, any, our, off, set, But, way, New, 000
    # Less common words length 3:	Dia, Tai, 235, Buy, Las, Kal, Tri, Jim, Onn, SEA, 469, SAI, NBE, 239, 430, 243, UGs, arc, thе, 009, Der, Yeh, LED, 380, 207, EWS, EFL, icy, 203, 730, SAF, Bad, JAT, 822, ski, LTI, AIs, Pte, Jos, 214

    # Print all words of length 4
    word_counts_4 = df.apply(lambda x: x.str.findall(r'\b\w{4}\b')).stack().explode().value_counts()
    print('Most common words length 4:\t' + ', '.join(word_counts_4.head(40).index))
    start_index = max(0, len(word_counts_4) // 3 - 20)
    end_index = min(len(word_counts_4), len(word_counts_4) // 3 + 20)
    print('Less common words length 4:\t' + ', '.join(word_counts_4.iloc[start_index:end_index].index))
    # Most common words length 4:	that, with, have, from, said, will, this, more, cent, year, 2023, also, been, Sign, time, were, they, film, This, over, last, your, when, free, like, 2024, news, read, than, With, into, only, Test, Also, Khan, team, them, made, what, some
    # Less common words length 4:	item, hide, ghar, hire, Jony, Road, NBFC, Lipa, PYQs, gaye, pink, 38th, cult, Evan, CBFC, slog, jets, Fast, slot, Raaz, Rory, Bath, Yuki, lady, Hota, teen, belt, Soni, CBDT, Rest, saga, bans, gear, Pele, Back, bath, Ivan, CGST, Mere, labs


def preprocess_column(column, min_count=0, apply_stemming=False):
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
    # Remove words whose length is more than 3 and have no syllable
    column = column.apply(lambda x: " ".join(x for x in str(x).split() if len(x) <= 3 or any(v in x for v in 'aeiou')))
    # Create dictionary of words and their frequency
    word_freq = pd.Series(' '.join(column).split()).value_counts()
    # Remove all elements that occur less than min_count times
    column = column.apply(lambda x: ' '.join([word for word in x.split() if word_freq[word] >= min_count]))
    if apply_stemming:
        # Stem words
        column = column.apply(lambda x: ' '.join(stem.stem(word) for word in x.split()))
    return column


def bag_of_words(df: DataFrame):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['headlines'] + ' ' + df['description'] + ' ' + df['content'])
    print(f"Shape: {X.shape}, Non-zero elements: {X.nnz}")
    return X, vectorizer.get_feature_names_out()


def process_and_evaluate_clustering(X, X_original, feature_names, method_name, n_clusters=5):
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='random')
    kmeans.fit(X)

    # Get the cluster labels
    labels = kmeans.labels_

    # Plot the documents per cluster
    plot_documents_per_cluster(labels, f"Document per cluster - {method_name} KMeans({n_clusters})")

    # Plot the cluster assignments
    plot_cluster_assignments(X, f"{method_name} KMeans({n_clusters})", labels)

    # Print cluster explanation table
    print_cluster_explanation_table(labels, X_original, feature_names)

    # Evaluate clustering
    evaluate_clustering(X, labels, method_name)

    return labels


def perform_clustering(algorithm, X, algorithm_name):
    # Fit the clustering algorithm
    algorithm.fit(X)
    labels = algorithm.labels_

    # Plot documents per cluster
    plot_documents_per_cluster(labels, f"Document per cluster - {algorithm_name}")

    # Plot cluster assignments
    plot_cluster_assignments(X, f"Cluster assignments - {algorithm_name}", labels)

    # Evaluate clustering
    evaluate_clustering(X, labels, algorithm_name)

    return labels


def plot_documents_per_cluster(cluster_assignments, title='Number of documents in each cluster', show=False):
    plt.clf()
    plt.hist(cluster_assignments, bins=len(set(cluster_assignments)))
    plt.xlabel('Cluster')
    plt.ylabel('Number of documents')
    plt.title(title)
    plt.savefig(f'Plots/{title}.png')
    if show:
        plt.show()
    plt.clf()


def plot_cluster_assignments(X, title, cluster_assignments=None, show=False):
    plt.clf()
    plt.figure(figsize=(10, 8))
    if cluster_assignments is not None:
        plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis')
    else:
        plt.scatter(X[:, 0], X[:, 1])
    plt.colorbar(label='Cluster label')
    plt.title(title)
    plt.savefig(f'Plots/{title}.png')
    if show:
        plt.show()


def print_cluster_explanation(labels, X_original, feature_names):
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


def print_cluster_explanation_table(labels, X_original, feature_names):
    start = -1 if -1 in labels else 0
    end = np.max(labels) + 1
    num_clusters = abs(end) + abs(start)

    # Calculate the mean TF-IDF score for each term in each cluster
    term_ratios = np.zeros((num_clusters, X_original.shape[1]))

    for i in range(start, end):
        # Find indices of documents in the cluster
        indices = np.where(labels == i)[0]
        # Aggregate TF-IDF scores by mean within the cluster
        term_ratios[i + abs(start), :] = np.mean(X_original[indices], axis=0)

    # Create table headers
    headers = [f'Cluster {i}' for i in range(start, end)]
    # Add cluster numbers to headers

    # Create a dictionary to hold data rows for the table
    data = {}

    # Populate the data dictionary with cluster information
    for i in range(num_clusters):
        top_terms = term_ratios[i].argsort()[::-1][:10]  # Get indices of top terms
        terms_str = [feature_names[ind] for ind in top_terms]
        data[f'Cluster {i}'] = terms_str

    # Print the table
    print(tabulate(data, headers=headers, tablefmt='grid'))

def tune_model(model, X, parameters):
    grid_search = GridSearchCV(model, parameters, scoring=silhouette_scorer, error_score='raise', verbose=0,
                               n_jobs=-1, cv=3)
    grid_search.fit(X)
    print(f"Best parameters for {model}: {grid_search.best_params_}")
    return grid_search.best_params_


def silhouette_scorer(estimator, X):
    estimator.fit(X)
    return silhouette_score(X, estimator.labels_)


def create_similarity_matrix(cluster_assignments):
    # Number of clusters
    num_clusters = np.max(cluster_assignments) + 1

    # Initialize the similarity matrix
    similarity_matrix = np.zeros((num_clusters, num_clusters))

    # Calculate the co-occurrence of clusters
    for i in range(len(cluster_assignments)):
        for j in range(i + 1, len(cluster_assignments)):
            cluster_i = cluster_assignments[i]
            cluster_j = cluster_assignments[j]
            if cluster_i == cluster_j:
                similarity_matrix[cluster_i][cluster_j] += 1
                similarity_matrix[cluster_j][cluster_i] += 1

    # Normalize the similarity matrix
    # To get a symmetric matrix with 1s on the diagonal
    for i in range(num_clusters):
        similarity_matrix[i][i] = len(cluster_assignments) // num_clusters
    similarity_matrix /= np.max(similarity_matrix)

    return similarity_matrix


def visualize_similarity_matrix(similarity_matrix, title='Similarity Matrix', show=False):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', linewidths=0.5, cbar=True)
    plt.title(title)
    plt.xlabel('Item')
    plt.ylabel('Item')
    if show:
        plt.show()
    plt.savefig(f'Plots/{title}.png')


def evaluate_clustering(X, labels, name='N/A'):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    print(f"Results for {name}: (Silhouette, {silhouette:.2f}), (Davies, {davies_bouldin:.2f}), (Calinski, {calinski_harabasz:.2f})")