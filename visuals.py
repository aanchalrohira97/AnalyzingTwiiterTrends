import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from wordcloud import WordCloud


def word_cloud(freq_dic):
    word_cloud = WordCloud(background_color="white").generate_from_frequencies(freq_dic)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def frequency_bar_chart(tags_freq, vocabulary, max=20):
    words_freq = [(word, tags_freq[0, idx]) for word, idx in vocabulary.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words = ["#" + word[0] for word in words_freq]
    freq = [freq[1] for freq in words_freq]
    plt.figure(figsize=(15, 10))
    x = np.arange(max)
    plt.barh(x, freq[:max], align='center')
    plt.yticks(x, words[:max])
    plt.ylabel('HashTags')
    plt.xlabel('Frequency')
    plt.title('Top Trending HashTags')
    plt.show()


def heat_map(X):
    sns.heatmap(X.todense()[:, np.random.randint(0, X.shape[1], 100)] == 0, vmin=0, vmax=1, cbar=False).set_title('Bag of Words')
    plt.show()


def clusters(data, labels):
    pca = PCA(n_components=2)
    T = pca.fit_transform(data)
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(T[idx, 0], T[idx, 1])
    plt.show()