import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.metrics import accuracy_score, hamming_loss


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
