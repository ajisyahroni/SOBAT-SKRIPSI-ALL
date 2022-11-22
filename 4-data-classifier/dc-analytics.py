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
           "SE_DM", ]

mm_attr = ["MM_2D",
           "MM_3D",
           "MM_MI",
           "MM_VIDEO",
           "MM_VFX",
           "MM_MG", ]

cn_attr = ["CN_SECURITY",
           "CN_INFRA_WEB_INET",
           "CN_NIRCABLE",
           "CN_NETWORK_PLAN_IMPL",
           "CN_IOT"]


'''FUNTIONS'''


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

# mm
y_mm = df[mm_attr]
X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
    vct, y_mm, test_size=.2, random_state=20)

# cn
y_cn = df[cn_attr]
X_train_cn, X_test_cn, y_train_cn, y_test_cn = train_test_split(
    vct, y_cn, test_size=.2, random_state=1998)


print(y_train_cn.sum(axis=0))
print(y_test_cn.sum(axis=0))
