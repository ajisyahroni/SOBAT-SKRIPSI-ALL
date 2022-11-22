import rich
from lib.clf_runner import ptclf
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import time
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
labels = [
    "SE_AI",
    "SE_WEB",
    "SE_MOBILE",
    "SE_DB",
    "SE_GAME",
    "SE_DM",

    "MM_2D",
    "MM_3D",
    "MM_MI",
    "MM_VIDEO",
    "MM_VFX",
    "MM_MG",

    "CN_SECURITY",
    "CN_INFRA_WEB_INET",
    "CN_NIRCABLE",
    "CN_NETWORK_PLAN_IMPL",
    "CN_IOT"
]


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])

                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


# preprocessing value
df = pd.read_csv('prepres/df_1_without_unused_word_func.csv')
corpus = df['Result'].apply(word_tokenize)

# model training
s_time = time.time()
model = Word2Vec(sentences=corpus, window=4, min_count=1, workers=4)
e_time = time.time()

# print(model.wv.similar_by_word('php'))

# vectorizing docs
vct = vectorize(corpus, model)

scaler = MinMaxScaler()
mvs_scl = scaler.fit_transform(vct)

X_train, X_test, y_train, y_test = train_test_split(
    mvs_scl, df[labels], test_size=0.2, random_state=1998)

'''clf'''
res = ptclf(X_train, y_train, X_test, y_test)
rich.print(f'\n{res}')
