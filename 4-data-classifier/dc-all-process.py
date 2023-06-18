from sklearn.feature_selection import VarianceThreshold
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from skmultilearn.model_selection import iterative_train_test_split
from sqlalchemy.types import *
from sqlalchemy import create_engine
import sqlite3
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd
import numpy as np

'''ABS VAR'''
se_attr = ["SE_AI",
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

mm_attr = []

cn_attr = []


'''FUNTIONS'''


def ptclf(X_train, y_train, X_test, y_test):
    # cc
    cc_clf = ClassifierChain(MultinomialNB(alpha=.6))
    cc_clf.fit(X_train, y_train)
    cc_pred = cc_clf.predict(X_test)

    # lp
    lp_clf = LabelPowerset(MultinomialNB(alpha=.7))
    lp_clf.fit(X_train, y_train)
    lp_pred = lp_clf.predict(X_test)

    # br
    br_clf = BinaryRelevance(MultinomialNB(alpha=.7))
    br_clf.fit(X_train, y_train)
    br_pred = br_clf.predict(X_test)

    result = {
        'Metode': ['Classifier Chain', 'Label Powerset', 'Binary Relevance'],
        'Hamming Loss': [hamming_loss(y_test, cc_pred), hamming_loss(y_test, lp_pred), hamming_loss(y_test, br_pred), ],
        'Akurasi': [accuracy_score(y_test, cc_pred), accuracy_score(y_test, lp_pred), accuracy_score(y_test, br_pred), ],
    }

    return pd.DataFrame(result)


'''MAIN PROCESS HERE'''
df = pd.read_csv('input/dp-out.csv')

# vectorizing each subfield
max_df = .9
min_df = .001
max_feat = 1400

tfidf = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_feat)
vct = tfidf.fit_transform(df['Result']).toarray()
features = tfidf.fit(df['Result']).get_feature_names_out()
matrix_vector = pd.DataFrame(vct, columns=features)

'''DATA SPLITTING'''
# se
y_se = df[se_attr]
X_train_se, X_test_se, y_train_se, y_test_se = train_test_split(
    vct, y_se, test_size=.2, random_state=1998)

'''CLASSIFIER'''
res_se = ptclf(X_train_se, y_train_se, X_test_se, y_test_se)


print(f'\n\nPEMROGRAMAN :\n {res_se}')
