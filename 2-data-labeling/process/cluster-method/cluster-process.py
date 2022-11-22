from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


df = pd.read_csv('input/da-output-22-11-11_13:37:53.csv')
corpus = df['Result']


# vectorizing
tfidf = TfidfVectorizer(max_df=.9, min_df=.001)
vector = tfidf.fit_transform(corpus)
features = tfidf.fit(corpus).get_feature_names_out()
# matrix_vector = pd.DataFrame(vector, columns=features)

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
    "MM_MOTION_GRAPHIC",

    "CN_SECURITY",
    "CN_INFRA_WEB_INET",
    "CN_NIRCABLE",
    "CN_NETWORK_PLAN_IMPL",
    "CN_IOT"
]
df['labeled'] = False
df[labels] = 0

# clustering KMEANS
kmeans = KMeans(n_clusters=17, random_state=42)
kmeans.fit(vector)
clusters = kmeans.labels_
df['cluster'] = clusters


df_subset = df[['eprintid', 'title', 'cluster', 'abstract']]
df_subset = df_subset.sort_values('cluster')
df_subset.to_csv('output/result.csv')
