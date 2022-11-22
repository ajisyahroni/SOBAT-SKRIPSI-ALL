import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from mlsmote import create_dataset, get_minority_instace, MLSMOTE
from skmultilearn.model_selection import iterative_train_test_split
"""
main function to use the MLSMOTE
"""

labels = [
    "SE_AI",
    "SE_WEB",
    "SE_MOBILE",
    # "SE_DB",
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

df = pd.read_csv('r_res_df_clean.csv')
corpus = df['Result']

# vct
tfidf = TfidfVectorizer(max_df=.9, min_df=.001, max_features=700)
vct = tfidf.fit_transform(corpus).toarray()
features = tfidf.fit(corpus).get_feature_names_out()
matrix_vector = pd.DataFrame(vct, columns=features)

X_train, y_train, X_test, y_test = iterative_train_test_split(
    matrix_vector.values, df[labels].values, test_size=.2)
'''classifiy'''


'''classifiy'''
# cc
cc_clf = ClassifierChain(MultinomialNB())
cc_clf.fit(X_train, y_train)
cc_pred = cc_clf.predict(X_test)

# lp
lp_clf = LabelPowerset(MultinomialNB())
lp_clf.fit(X_train, y_train)
lp_pred = lp_clf.predict(X_test)

# br
br_clf = LabelPowerset(MultinomialNB())
br_clf.fit(X_train, y_train)
br_pred = br_clf.predict(X_test)


result = {
    'Metode': ['Classifier Chain', 'Label Powerset', 'Binary Relevance'],
    'Hamming Loss': [hamming_loss(y_test, cc_pred), hamming_loss(y_test, lp_pred), hamming_loss(y_test, br_pred), ],
    'Akurasi': [accuracy_score(y_test, cc_pred), accuracy_score(y_test, lp_pred), accuracy_score(y_test, br_pred), ],
}

print(f'\n{pd.DataFrame(result)}\n')
