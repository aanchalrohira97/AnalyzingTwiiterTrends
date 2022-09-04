import re

import nltk
from nltk.tokenize import word_tokenize

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''

# pre-processing function to remove stop words, stemming, and lemmetization

maps = {"amzn": "amazon", "msft": "microsoft", "fb": "facebook", "aapl": "apple", "tsla": "tesla", "ntflx": "netflix",
        "googl": "google", "twtr": "twitter"}


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # spelling correction
    # text = TextBlob(text).correct()

    # Tokenize (convert from string to list)
    lst_text = word_tokenize(text)

    # should be english word only
    lst_text = [word for word in lst_text if word.isalpha()]

    out = []
    for word in lst_text:
        if word.isalpha():
            if word in maps:
                out.append(maps[word])
            else:
                out.append(word)
    lst_text = out

    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    # Lemmatization (convert the word into root word)
    if flg_lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word, pos='v') for word in lst_text]

        # Stemming (remove -ing, -ly, ...)
    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word, to_lowercase=False) for word in lst_text]
    # back to string from list
    text = " ".join(lst_text)

    return text


def hashTags(tweet):
    out = ""
    for t in tweet.split():
        if t.startswith("#"):
            out += t.lower() + " "
    return out