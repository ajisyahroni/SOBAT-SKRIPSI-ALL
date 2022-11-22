

import rich
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
from lib.prep import TextPrep, TextPrepVis
from lib.label_inspector import LabelInspector
import pandas as pd
import numpy as np
from lib.clf_runner import ptclf

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


# eksperimen


df = pd.read_csv('prepres/df_5_improve_unused_word_func.csv')
corpus = df['Result']

'''idf'''
tfidf = TfidfVectorizer(max_df=.9, min_df=.001, max_features=700)
vct = tfidf.fit_transform(corpus).toarray()
features = tfidf.fit(corpus).get_feature_names_out()
matrix_vector = pd.DataFrame(vct, columns=features)

'''data split'''
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(
    matrix_vector, y, test_size=0.2, random_state=1998)

'''impl feature selection by variance treshold'''
constant_filter = VarianceThreshold(threshold=0.0002)
constant_filter.fit(X_train)
feature_list = X_train[X_train.columns[constant_filter.get_support(
    indices=True)]]

x_train_filter = constant_filter.transform(X_train)
x_test_filter = constant_filter.transform(X_test)


'''clf'''
res = ptclf(x_train_filter, y_train, x_test_filter, y_test)
rich.print(f'\n{res}')
