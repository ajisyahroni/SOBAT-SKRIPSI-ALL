import rich
from lib.clf_runner import ptclf
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
from lib.mlsmote import create_dataset, get_minority_instace, MLSMOTE
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
# X, y = create_dataset()  # Creating a Dataframe

# Getting minority instance of that datframe
# X_sub, y_sub = get_minority_instace(X, y)

# Applying MLSMOTE to augment the dataframe
# X_res, y_res = MLSMOTE(X_sub, y_sub, 100)

# my dateset
df = pd.read_csv('prepres/df_1_without_unused_word_func.csv')
corpus = df['Result']
tfidf = TfidfVectorizer(max_df=.9, min_df=.001, max_features=700)
vct = tfidf.fit_transform(corpus).toarray()
features = tfidf.fit(corpus).get_feature_names_out()
matrix_vector = pd.DataFrame(vct, columns=features)


aug = 500
X_sub, y_sub = get_minority_instace(matrix_vector, df[labels])
X_res, y_res = MLSMOTE(X_sub, y_sub, aug)
matrix_vector_smoted = pd.concat(
    [matrix_vector, X_res], ignore_index=True, sort=False)
y_smoted = pd.concat([df[labels], y_res], ignore_index=True, sort=False)


# RUS
_df = pd.read_csv('prepres/df_8_rus.csv')
_corpus = _df['Result']
_tfidf = TfidfVectorizer(max_df=.9, min_df=.001, max_features=700)
_vct = _tfidf.fit_transform(_corpus).toarray()
_features = _tfidf.fit(_corpus).get_feature_names_out()
_matrix_vector = pd.DataFrame(_vct, columns=_features)
_y = _df[labels]

all_x = pd.concat([_matrix_vector, X_res], ignore_index=True, sort=False)
all_y = pd.concat([_y, y_smoted], ignore_index=True, sort=False)

# scaler remove fucking negative valeu
# scaler = MinMaxScaler()
# mvs_scl = scaler.fit_transform(matrix_vector_smoted)

xx = pd.concat([all_x,all_y], axis=1)
xx = xx.fillna(0)

xx.to_csv('12_tmp.csv')
scaler = MinMaxScaler()
mvs_scl = scaler.fit_transform(xx)
    
X_train, X_test, y_train, y_test = train_test_split(
        mvs_scl, xx[labels], test_size=0.2, random_state=1998)

res = ptclf(X_train, y_train, X_test, y_test)
rich.print(res)