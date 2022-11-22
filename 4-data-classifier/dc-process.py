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


# vectorizing each subfield
max_df = .9
min_df = .001
max_feat = 200

# se
df_se = pd.read_csv('input/df_se_clean.csv')
tfidf_se = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_feat)
vct_se = tfidf_se.fit_transform(df_se['Result']).toarray()
features_se = tfidf_se.fit(df_se['Result']).get_feature_names_out()
matrix_vector_se = pd.DataFrame(vct_se, columns=features_se)

# mm
df_mm = pd.read_csv('input/df_mm_clean.csv')
tfidf_mm = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_feat)
vct_mm = tfidf_mm.fit_transform(df_mm['Result']).toarray()
features_mm = tfidf_mm.fit(df_mm['Result']).get_feature_names_out()
matrix_vector_mm = pd.DataFrame(vct_mm, columns=features_mm)

# cn
df_cn = pd.read_csv('input/df_cn_clean.csv')
tfidf_cn = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_feat)
vct_cn = tfidf_cn.fit_transform(df_cn['Result']).toarray()
features_cn = tfidf_cn.fit(df_cn['Result']).get_feature_names_out()
matrix_vector_cn = pd.DataFrame(vct_cn, columns=features_cn)



# vct all 
# tfidf = TfidfVectorizer(max_df=0., min_df=min_df, max_features=max_feat)

'''DATA SPLITTING'''
# var
test_size = .2
random_val = 6
# 6
# se
y_se = df_se[se_attr]
X_train_se, X_test_se, y_train_se, y_test_se = train_test_split(
    vct_se, y_se, test_size=test_size, random_state=random_val)


# mm
y_mm = df_mm[mm_attr]
X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
    vct_mm, y_mm, test_size=test_size, random_state=random_val)


# cn
y_cn = df_cn[cn_attr]
X_train_cn, X_test_cn, y_train_cn, y_test_cn = train_test_split(
    vct_cn, y_cn, test_size=test_size, random_state=random_val)


'''CLASSIFIER'''
res_se = ptclf(X_train_se, y_train_se, X_test_se, y_test_se)
res_mm = ptclf(X_train_mm, y_train_mm, X_test_mm, y_test_mm)
res_cn = ptclf(X_train_cn, y_train_cn, X_test_cn, y_test_cn)

print(f'\n\nse : {res_se}')
print(f'\n\nmm : {res_mm}')
print(f'\n\ncn : {res_cn}')