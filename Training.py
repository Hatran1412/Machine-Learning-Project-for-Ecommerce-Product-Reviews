import pandas as pd
import numpy as np
from joblib import load, dump
from copy import deepcopy
from statistics import mean

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counte

df = pd.read_csv('../output/Features.csv')
data_split = pd.crosstab(df['product'],df['label'])

def building_training_data(df):
    A = df[df['label']==1]
    A.loc[df['label']==1,'join'] = 'j'
    B = df[df['label']==0]
    B.loc[df['label']==0,'join'] = 'j'
    trainset1 = pd.merge(A,B,how='outer',on='join')
    trainset2 = pd.merge(B,A,how='outer',on ='join')

    trainset = pd.merge(trainset1,trainset2,how='outer')
    return trainset

product_list = df['product'].unique()
data_stack = []
for product in product_list:
    temp = deepcopy(df[df['product']==product].iloc[:,2:])
    build_data = building_training_data(temp)
    print(product, len(temp), len(build_data))
    build_data.drop(columns = ['join','label_y'],inplace=True)
    data = build_data.iloc[:,1:]
    data['target'] = build_data.iloc[:,0]
    data_stack.append(data)

train = pd.concat(data_stack).reset_index(drop = True)

X = train.iloc[:,:-1].values
y = train.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,shuffle = True, stratify = y) 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
print("Training Accuracy\n", accuracy_score(y_train,classifier.predict(X_train)))
print("Test Accuracy\n", accuracy_score(y_test,classifier.predict(X_test)))

print('CLASSIFICATION REPORT')
print("Training\n", classification_report(y_train,classifier.predict(X_train)))
print("Test \n", classification_report(y_test,classifier.predict(X_test)))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

print("Training Accuracy\n", accuracy_score(y_train,classifier.predict(X_train)))
print("Test Accuracy\n", accuracy_score(y_test,classifier.predict(X_test)))

print('CLASSIFICATION REPORT')
print("Training\n", classification_report(y_train,classifier.predict(X_train)))
print("Test \n", classification_report(y_test,classifier.predict(X_test)))



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=50, n_jobs = -1, oob_score = True,random_state=42)
classifier.fit(X_train,y_train)

print("Training Accuracy\n", accuracy_score(y_train,classifier.predict(X_train)))
print("Test Accuracy\n", accuracy_score(y_test,classifier.predict(X_test)))

print('CLASSIFICATION REPORT')
print("Training\n", classification_report(y_train,classifier.predict(X_train)))
print("Test \n", classification_report(y_test,classifier.predict(X_test)))

print("Test\nConfusion Matrix: \n", confusion_matrix(y_test, classifier.predict(X_test)))


## Score of the training dataset obtained using an out-of-bag estimate. This attribute exists only when oob_score is True.
classifier.oob_score_

feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = train.iloc[:,:-1].columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

dump(classifier, '../output/randomforest.joblib', compress = 2)

#PART 2. Model Ranking Metric

classifier = load('../output/randomforest.joblib')

product_list = df['product'].unique()
df['win']=0
df['lose']=0
df['review_score'] = 0.0
df.reset_index(inplace = True, drop = True)


def score_giver(C,D):
    E = pd.merge(C,D,how='outer',on='j')
    E.drop(columns=['j'],inplace = True)
    q= classifier.predict(E.values)
    return Counter(q)

for product in product_list:
    data = df[df['product']==product]
    for indx in data.index:
        review = df.iloc[indx, 3:-3]
        review['j'] = 'jn'
        C = pd.DataFrame([review])
        D = data[data.index!=indx].iloc[:,3:-3]
        D['j'] = 'jn'
        score = score_giver(C,D)
        df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
        df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
        df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0

df = df.sort_values(by = ['product','review_score'], ascending = False)

r_accuracy =[]
for product in product_list:
    x = data_split[data_split.index == product][1][0]
    number_of_1_in_x = Counter(df[df['product']==product].iloc[:x, ]['label']).get(1)
    rank_accuracy = float(number_of_1_in_x*1.0 / x*1.0)
    print("Product: {} | Rank Accuracy: {}".format(product, rank_accuracy))
    r_accuracy.append(rank_accuracy)
print("Mean Rank Accuracy: {}".format(mean(r_accuracy)))

df.iloc[:, [0,1,-1]].to_csv('../output/train_ranked_output.csv',index = False)
