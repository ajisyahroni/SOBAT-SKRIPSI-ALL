''' 
    DATA ACQUISITION AND PREPARATION : 
    - remove duplicate data 
    - remove unused data
    - turn splitted data into one data
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import os


# 2021 (560)
# 2020 (466)
# 2019 (340)


# load the datas

df_2019 = pd.read_json('input/id_if_2019_all.json')
df_2020 = pd.read_json('input/id_if_2020_all.json')
df_2021 = pd.read_json('input/id_if_2021_all.json')

# reassign attrs
attrs = ['title', 'keywords', 'abstract', 'eprintid', 'uri', "creators"]
df_2019 = df_2019[attrs]
df_2020 = df_2020[attrs]
df_2021 = df_2021[attrs]

# assign year to all data only
df_2019['year'] = 2019
df_2020['year'] = 2020
df_2021['year'] = 2021
all_data = [df_2019, df_2020, df_2021]
df_all: pd.DataFrame = pd.concat(all_data, ignore_index=True, sort=False)

'''REMOVE DUPLICATE DATA'''
# remove duplicate eprint id
print(f'before drop_dup : {df_all.shape[0]}')
df_all = df_all.drop_duplicates(subset=['title'], keep='first')
df_all = df_all.drop_duplicates(subset=['abstract'], keep='first')
print(f'after drop_dup : {df_all.shape[0]}')

# remove specified topics
removed_docs = 'overclock|overclocking|berbasis desktop'
df_to_remove = df_all['abstract'].str.contains(removed_docs, case=False)
print(df_all.loc[df_to_remove])

df_all = df_all.loc[~df_to_remove]


timestamp = '{:%y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
filename = f'output/da-output-{timestamp}.csv'
df_all.to_csv(filename)
