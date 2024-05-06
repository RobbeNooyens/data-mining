from pandas import DataFrame


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


def explain(kmeans_model, terms):
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    for i in range(len(order_centroids)):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print('--------------------------------')