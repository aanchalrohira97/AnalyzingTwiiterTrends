from sklearn import feature_extraction
from sklearn.cluster import KMeans


def get_bag_oof_words(corpus, vect='count', ngram_range=(1, 1)):
    # tokenization and using TF-IDF as scoring mechanism
    vectorizer = feature_extraction.text.CountVectorizer(ngram_range=ngram_range)
    if vect == 'tfidf':
        vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    return vectorizer, vectorizer.fit_transform(corpus)


def cluster_kmeans(training_data, n_clusters=3):
    return KMeans(n_clusters=n_clusters, random_state=0).fit(training_data)


def binary_vectorizer(corpus, ngram_range=(1,1)):
    vectorizer = feature_extraction.text.CountVectorizer(binary=True, ngram_range=ngram_range)
    return vectorizer, vectorizer.fit_transform(corpus)