import pandas as pd
import numpy as np


df = pd.read_csv('../output/final-all-dl-out.csv')
abs_attr = ['title', 'keywords', 'abstract',
            'eprintid', 'uri', 'creators', 'year']

# assigning subfield
df['subfield'] = np.NaN


'''CN'''
cn_attr = ["CN_SECURITY",
           "CN_INFRA_WEB_INET",
           "CN_NIRCABLE",
           "CN_NETWORK_PLAN_IMPL",
           "CN_IOT"]

cn_con = (df['CN_SECURITY'] == 1) | (df['CN_INFRA_WEB_INET'] == 1) | (
    df["CN_NIRCABLE"] == 1) | (df["CN_NETWORK_PLAN_IMPL"] == 1) | (df["CN_IOT"] == 1)
df_cn = pd.concat([df.loc[cn_con]])
df_cn['subfield'] = 'CN'
df_cn = df_cn[[*abs_attr, *cn_attr]]

'''MM'''
mm_attr = ["MM_2D",
           "MM_3D",
           "MM_MI",
           "MM_VIDEO",
           "MM_VFX",
           "MM_MG", ]
mm_con = (df['MM_2D'] == 1) | (df['MM_3D'] == 1) | (
    df["MM_MI"] == 1) | (df["MM_VIDEO"] == 1) | (df["MM_VFX"] == 1) | (df["MM_MG"] == 1)
df_mm = pd.concat([df.loc[mm_con]])
df_mm['subfield'] = 'MM'
df_mm = df_mm[[*abs_attr, *mm_attr]]

'''SE'''
se_attr = ["SE_AI",
           "SE_WEB",
           "SE_MOBILE",
           "SE_DB",
           "SE_GAME",
           "SE_DM", ]
se_con = (df['SE_AI'] == 1) | (df['SE_WEB'] == 1) | (
    df["SE_MOBILE"] == 1) | (df["SE_DB"] == 1) | (df["SE_GAME"] == 1) | (df["SE_DM"] == 1)
df_se = pd.concat([df.loc[se_con]])
df_se['subfield'] = 'SE'
df_se = df_se[[*abs_attr, *se_attr]]

# outputing into separate files
df_cn.to_csv('../output/dl_cn.csv')
df_mm.to_csv('../output/dl_mm.csv')
df_se.to_csv('../output/dl_se.csv')


# df_all = pd.concat([df_cn, df_mm, df_se], ignore_index=True, sort=False)
