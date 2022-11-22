import datetime
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
import pandas as pd
import numpy
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


df = pd.read_csv('input/da-output-22-11-11_13:37:53.csv')
df[labels] = 0

hehe = [
    # se
    {
        'code': 'SE_AI',
        'keywords': 'sistem penunjang keputusan|forward chaining|artificial intelligence|SISTEM PAKAR|CITRA DIGITAL|DEEP LEARNING|IMAGE PROCESSING|JARINGAN SYARAF TIRUAN'
    },
    {
        'code': 'SE_WEB',
        'keywords': 'PHP|website'
    },
    {
        'code': 'SE_MOBILE',
        'keywords': 'android|mobile'
    },
    {
        'code': 'SE_DB',
        'keywords': 'mysql|my sql'
    },
    {
        'code': 'SE_GAME',
        'keywords': 'game'
    },
    {
        'code': 'SE_DM',
        'keywords': 'data mining|apriori|klasifikasi|sentimen|SISTEM REKOMENDASI'
    },

    # multimedia
    {
        'code': "MM_2D",
        'keywords': '2D|2 dimensi'
    },
    {
        'code': "MM_3D",
        'keywords': '3D| 3 dimensi'
    },
    {
        'code': "MM_MI",
        'keywords': 'media interaktif|augmented reality|virtual reality'
    },
    {
        'code': "MM_VIDEO",
        'keywords': 'iklan|promosi'
    },
    {
        'code': "MM_VFX",
        'keywords': 'vfx|visual effect'
    },
    {
        'code': "MM_MG",
        'keywords': 'motion graphic'
    },

    # jaringan
    {
        'code': "CN_SECURITY",
        'keywords': 'keamanan jaringan|kriptografi|STEGANOGRAFI|firewall',
    },
    {
        'code': "CN_INFRA_WEB_INET",
        'keywords': 'apache|nginx|infrastuktur|httperf',
    },
    {
        'code': "CN_NIRCABLE",
        'keywords': 'hotspot|wireless|MIKROTIK',
    },
    {
        'code': "CN_NETWORK_PLAN_IMPL",
        'keywords': 'perancangan jaringan|implementasi jaringan|penerapan JARINGAN|IMPLEMENTASI MANAJEMEN BANDWIDTH',
    },
    {
        'code': "CN_IOT",
        'keywords': 'internet of things|internet of thing|iot',
    },
]

for it in hehe:
    df_db = df['abstract'].str.contains(it['keywords'], case=False)
    df.loc[df_db, it['code']] = 1


timestamp = '{:%y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
filename = f'output/km-out-{timestamp}.csv'
df.to_csv(filename)
