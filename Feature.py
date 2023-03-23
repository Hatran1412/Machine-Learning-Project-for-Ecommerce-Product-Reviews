import pandas as pd
import numpy as np
from copy import deepcopy

import sys
sys.path.append('./utils')
from utils import review_feature
rf = review_feature()

from pandas_profiling import ProfileReport

df = pd.read_csv('../output/Preprocessed_Reviews.csv').sort_values(by = ['product'], ignore_index = True)

## Add Feature Columns
df['Rn'] = 0.0
df['Rp'] = 0.0
df['Rs'] = 0.0
df['Rc'] = 0.0
df['Rd'] = 0.0
df['Rsc'] = 0.0

product_list = df['product'].unique()

for product in product_list:
    data = df[df['product']==product]
    unique_bag = set()
    for review in data['answer_option']:
        review = review.lower()
        words = review.split()
        unique_bag = unique_bag.union(set(words))

    for indx in data.index:
        review = data.at[indx, 'answer_option']
        df.at[indx, 'Rp'] = rf.polarity_sentiment(review)
        df.at[indx, 'Rs'] = rf.subjectivity_sentiment(review)
        df.at[indx, 'Rd'] = rf.service_tag(review)
        df.at[indx, 'Rsc'] = rf.slang_emoji_polarity_compoundscore(review)
        df.at[indx, 'Rc'] = float(len(set(review.split()))) / float(len(unique_bag))

    df.loc[df['product']==product, 'Rn'] = rf.noun_score(data['answer_option'].values).values

df.to_csv('../output/Features.csv',index = False)

profile = ProfileReport(df)

import os
try:
    os.mkdir('../output/profiler')
except:
    pass
profile.to_file(output_file="../output/profiler/feature_analysis.html")