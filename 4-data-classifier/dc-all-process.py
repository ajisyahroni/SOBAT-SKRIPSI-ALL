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

'''CLASSIFIER'''
res_se = ptclf(X_train_se, y_train_se, X_test_se, y_test_se)
res_mm = ptclf(X_train_mm, y_train_mm, X_test_mm, y_test_mm)
res_cn = ptclf(X_train_cn, y_train_cn, X_test_cn, y_test_cn)

print(f'\n\nPEMROGRAMAN :\n {res_se}')
print(f'\n\nMULTIMEDIA :\n {res_mm}')
print(f'\n\nJARINGAN :\n {res_cn}')


'''COLUMNS ADJUSTMENT'''
# se
cc_clf_se = ClassifierChain(MultinomialNB(alpha=.6))
cc_clf_se.fit(X_train_se, y_train_se)
cc_best_se = cc_clf_se.predict(matrix_vector)
df_res_se = pd.DataFrame(cc_best_se.toarray(), columns=se_attr)

df_res_se['title'] = df['title']
df_res_se['abstract'] = df['abstract']
df_res_se['creators'] = df['creators']
df_res_se['eprintid'] = df['eprintid']
df_res_se['uri'] = df['uri']
df_res_se['keywords'] = df['keywords']
df_res_se['year'] = df['year']


# mm
cc_clf_mm = ClassifierChain(MultinomialNB(alpha=.6))
cc_clf_mm.fit(X_train_mm, y_train_mm)
cc_best_mm = cc_clf_mm.predict(matrix_vector)
df_res_mm = pd.DataFrame(cc_best_mm.toarray(), columns=mm_attr)

df_res_mm['title'] = df['title']
df_res_mm['abstract'] = df['abstract']
df_res_mm['creators'] = df['creators']
df_res_mm['eprintid'] = df['eprintid']
df_res_mm['uri'] = df['uri']
df_res_mm['keywords'] = df['keywords']
df_res_mm['year'] = df['year']


# cn
cc_clf_cn = ClassifierChain(MultinomialNB(alpha=.6))
cc_clf_cn.fit(X_train_cn, y_train_cn)
cc_best_cn = cc_clf_cn.predict(matrix_vector)
df_res_cn = pd.DataFrame(cc_best_cn.toarray(), columns=cn_attr)

df_res_cn['title'] = df['title']
df_res_cn['abstract'] = df['abstract']
df_res_cn['creators'] = df['creators']
df_res_cn['eprintid'] = df['eprintid']
df_res_cn['uri'] = df['uri']
df_res_cn['keywords'] = df['keywords']
df_res_cn['year'] = df['year']


'''SUBFIELD'''
# se
se_con = (df_res_se['SE_AI'] == 1) | (df_res_se['SE_WEB'] == 1) | (
    df_res_se["SE_MOBILE"] == 1) | (df_res_se["SE_DB"] == 1) | (df_res_se["SE_GAME"] == 1) | (df_res_se["SE_DM"] == 1)

df_res_se = pd.concat([df_res_se.loc[se_con]])
df_res_se['subfield'] = 'SE'
# mm
mm_con = (df_res_mm['MM_2D'] == 1) | (df_res_mm['MM_3D'] == 1) | (
    df_res_mm["MM_MI"] == 1) | (df_res_mm["MM_VIDEO"] == 1) | (df_res_mm["MM_VFX"] == 1) | (df_res_mm["MM_MG"] == 1)
df_res_mm = pd.concat([df_res_mm.loc[mm_con]])
df_res_mm['subfield'] = 'MM'
# cn
cn_con = (df_res_cn['CN_SECURITY'] == 1) | (df_res_cn['CN_INFRA_WEB_INET'] == 1) | (
    df_res_cn["CN_NIRCABLE"] == 1) | (df_res_cn["CN_NETWORK_PLAN_IMPL"] == 1) | (df_res_cn["CN_IOT"] == 1)
df_res_cn = pd.concat([df_res_cn.loc[cn_con]])
df_res_cn['subfield'] = 'CN'


# insert into db
con = sqlite3.connect('output/db.sqlite')
df_res_se.to_sql(name='SE', con=con, if_exists='replace')
df_res_mm.to_sql(name='MM', con=con, if_exists='replace')
df_res_cn.to_sql(name='CN', con=con, if_exists='replace')
