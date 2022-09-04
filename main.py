import nltk
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

import models
import prep
import visuals

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
lst_stopwords = nltk.corpus.stopwords.words("english")

df = pd.read_csv('tweets.csv', delimiter=';')
df = df.drop('sentiment', axis=1)
df = df.drop('id', axis=1)
df = df.drop('created_at', axis=1)
unicode_clean_text = []
unicode_tags = []


def text_preprocess():
    # creating a new clean text column for each text
    df["text_clean"] = df["text"].apply(
        lambda x: prep.utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
    df.head()
    # extracting and storing hashtags from each tweet
    df["tags"] = df["text"].apply(lambda x: prep.hashTags(x))
    global unicode_clean_text
    global unicode_tags
    unicode_clean_text = df["text_clean"].values.astype('U')
    unicode_tags = df["tags"].values.astype('U')


def create_frequency_dictionary(corpus, ngram_range=(1, 1)):
    vectorizer, X = models.get_bag_oof_words(corpus, vect='tfidf', ngram_range=ngram_range)
    freq = X.toarray().sum(axis=0)
    freq_dic = {}
    for key, val in vectorizer.vocabulary_.items():
        freq_dic[key] = freq[val]
    return freq_dic


def word_cloud_of_cleaned_text(text, ngram_range=(1, 1)):
    freq_dic = create_frequency_dictionary(text, ngram_range)
    visuals.word_cloud(freq_dic)


def frequency_bar_chart_of_trending_hashtags(text, ngram_range=(1, 1)):
    vectorizer, X = models.get_bag_oof_words([t for t in text if t is not None and t != "nan"], ngram_range=ngram_range)
    visuals.frequency_bar_chart(X.sum(axis=0), vectorizer.vocabulary_, max=30)


def build_model(n_clusters=2, ngram_range=(1, 1)):
    vectorizer, X = models.get_bag_oof_words(unicode_clean_text, ngram_range=ngram_range)
    visuals.heat_map(X)
    kmeans = models.cluster_kmeans(X, n_clusters=n_clusters)
    visuals.clusters(X.toarray(), kmeans.labels_)
    for label in np.unique(kmeans.labels_):
        word_cloud_of_cleaned_text(unicode_clean_text[kmeans.labels_ == label], ngram_range)
        frequency_bar_chart_of_trending_hashtags(unicode_tags[kmeans.labels_ == label], ngram_range)


def tweets_analytics():
    text_preprocess()
    word_cloud_of_cleaned_text(unicode_clean_text, ngram_range=(1, 1))
    frequency_bar_chart_of_trending_hashtags(unicode_tags, ngram_range=(1, 1))
    build_model()


def prepare(ngram_range=(1, 1)):
    vectorizer, X = models.binary_vectorizer([t for t in unicode_clean_text if t is not None and t != "nan"],
                                             ngram_range=ngram_range)
    words_freq = [(word, idx) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1])
    words_freq = [x[0] for x in words_freq]
    out = pd.DataFrame(data=X.toarray(), columns=words_freq)
    return out


def mine_association_rules(metric='confidence', min_support=0.02, min_threshold=0.7, ngram_range=(1, 1)):
    hot_encoded_df = prepare(ngram_range=ngram_range)
    frq_items = fpgrowth(hot_encoded_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frq_items, metric=metric, min_threshold=min_threshold)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv('rules.csv', index=False)
    print(rules[['antecedents', 'consequents']])


tweets_analytics()
mine_association_rules(ngram_range=(1, 1))