from lib.prep import TextPrep

import pandas as pd
import numpy as np

df_se = pd.read_csv('input/dl_se.csv')
df_mm = pd.read_csv('input/dl_mm.csv')
df_cn = pd.read_csv('input/dl_cn.csv')

print('text prep se')
df_se_clean = TextPrep.fit(dataset=df_se, column_name='abstract', lang='id')
df_se_clean.to_csv("output/df_se_clean.csv")

print('text prep mm')
df_mm_clean = TextPrep.fit(dataset=df_mm, column_name='abstract', lang='id')
df_mm_clean.to_csv("output/df_mm_clean.csv")

print('text prep cn')
df_cn_clean = TextPrep.fit(dataset=df_cn, column_name='abstract', lang='id')
df_cn_clean.to_csv("output/df_cn_clean.csv")
