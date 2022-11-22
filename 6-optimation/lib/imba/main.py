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
# X, y = create_dataset()  # Creating a Dataframe

# Getting minority instance of that datframe
# X_sub, y_sub = get_minority_instace(X, y)

# Applying MLSMOTE to augment the dataframe
# X_res, y_res = MLSMOTE(X_sub, y_sub, 100)

# my dateset
df = pd.read_csv('df_1_without_unused_word_func.csv')
corpus = df['Result']
tfidf = TfidfVectorizer(max_df=.9, min_df=.001, max_features=700)
vct = tfidf.fit_transform(corpus).toarray()
features = tfidf.fit(corpus).get_feature_names_out()
matrix_vector = pd.DataFrame(vct, columns=features)


X_sub, y_sub = get_minority_instace(matrix_vector, df[labels])
X_res, y_res = MLSMOTE(X_sub, y_sub, 0)


# penggabungan dengan data oversampling
before_os = {
    'x': len(matrix_vector),
    'y': len(df[labels])
}
print(before_os)
print(f'sbelum resampling : {before_os}')
matrix_vector_smoted = pd.concat(
    [matrix_vector, X_res], ignore_index=True, sort=False)
y_smoted = pd.concat([df[labels], y_res], ignore_index=True, sort=False)
after_os = {
    'x': len(matrix_vector_smoted),
    'y': len(y_smoted)
}
print(f'setelah resampling : {after_os}')


# scaler remove fucking negative valeu

scaler = MinMaxScaler()
mvs_scl = scaler.fit_transform(matrix_vector_smoted)

X_train, X_test, y_train, y_test = train_test_split(
    mvs_scl, y_smoted, test_size=0.2, random_state=1998)
# X_train, y_train, X_test, y_test = iterative_train_test_split(
#     mvs_scl, y_smoted.values, test_size=0.2)

# fig = plt.figure(figsize=(20, 20))
# (col_1, col_2) = fig.subplots(ncols=2, nrows=1)
# g1 = sns.barplot(x=y_train.sum(axis=0), y=labels, ax=col_1)
# g2 = sns.barplot(x=y_test.sum(axis=0), y=labels, ax=col_2)
# g1.set_title("jumlah data train per topik")
# g2.set_title("jumlah data test per topik")
# plt.show()


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
