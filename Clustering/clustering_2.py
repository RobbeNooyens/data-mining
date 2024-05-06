import pandas as pd
from nltk.corpus import words

df = pd.read_excel('assignment3_articles.xlsx').drop(columns=['Unnamed: 0'])

documents = df['description'].values.astype('U')

# replace all non-alphabetic characters with a space
import re

documents = [re.sub(r'[^a-zA-Z]', ' ', document) for document in documents]

# Replace all spaces with a single space
documents = [re.sub(r'\s+', ' ', document) for document in documents]

english_vocab = set(words.words())
filtered_documents = []
for document in documents:
    words_list = document.split()
    filtered_words = [word for word in words_list if word in english_vocab]
    filtered_documents.append(' '.join(filtered_words))
documents = filtered_documents


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.5)
X = vectorizer.fit_transform(documents)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=0)
kmeans.fit(X)

df['cluster'] = kmeans.labels_

print(df['cluster'].value_counts())

