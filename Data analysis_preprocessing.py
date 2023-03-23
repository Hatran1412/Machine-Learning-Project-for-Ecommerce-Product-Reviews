import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7, 6

from copy import deepcopy
import seaborn as sns

import sys
sys.path.append('./utils')
from utils import review_feature
rf = review_feature()
%matplotlib inline

df = pd.read_csv('../data/train.csv')
label_analysis = pd.crosstab(df['product'],df['label'], margins='All')

analysis = label_analysis.reset_index()
analysis.columns = ['product','not info', 'info', 'All']
analysis.iloc[:-1].plot(x="product", y=["not info", "info"], kind="bar")

df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))

checklen = []
for i in range(5,50, 5):
    checklen.append(len(df[ (df['review_len']>=i-5) & (df['review_len']<i)]))

index = np.arange(len(checklen))
plt.bar(index, checklen)
plt.xlabel('Length of a Review', fontsize=15)
plt.ylabel('No. of Reviews', fontsize=15)
plt.xticks(index, range(5,50,5), fontsize=15, rotation=30)
plt.title('Review Survey Length Analysis')
plt.show()

#Stage1: Language Detection

bad_reviews = []
for indx in df.index:
    review = df.at[indx, 'answer_option']
    try:
        b = rf.language_detection(review)
        if b == 'hi' or b == 'mr':
            bad_reviews.append(indx)
    except:
        bad_reviews.append(indx)
        print("Language exception for:", review)

df[df.index.isin(bad_reviews)]

df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)

#Stage 2: Gibberish Review
bad_reviews = []
for indx in df.index:
    review = df.at[indx, 'answer_option']
    if rf.gibberish_detection(review, prefix_path = 'utils'):
        bad_reviews.append(indx)

df[df.index.isin(bad_reviews)]
df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)

#Stage 3: Profanity Detection

bad_reviews = []
for indx in df.index:
    review = df.at[indx, 'answer_option']
    if rf.english_swear_check(review) or rf.hindi_swear_check(review):
        bad_reviews.append(indx)

df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)

#Stage 4: Spelling Correction (Optional Stage not that necessary)

bad_reviews = []
for indx in df.index:
    review = df.at[indx, 'answer_option']
    if rf.competitive_brand_tag(review):
        bad_reviews.append(indx)

df = df[~df.index.isin(bad_reviews)].reset_index(drop = True)

import os
try:
    os.mkdir('../output')
except:
    pass

df.to_csv('../output/Preprocessed_Reviews.csv',index = False)

