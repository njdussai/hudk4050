
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

cols = [
        'SCHOOL',
        'GRADE',
        'CODER',
        'Gender',
        'OBSNUM',
        'totalobs-forsession',
        'Activity',
        'ONTASK',
        'TRANSITIONS',
        'FORMATchanges',
        'Obsv/act',
        'Transitions/Durations',
        'Total Time'
        ]

def prepare_data(df):
    dfx = df.loc[:, df.columns != 'ONTASK']
    Xs = pd.get_dummies(dfx, columns = ['SCHOOL', 'GRADE', 'CODER', 'Activity'])#, drop_first=True)

    Y = df['ONTASK'].replace(to_replace = ['Y', 'N'], value = [1, 0])#.to_numpy()
    return Xs, Y

#Import data
df = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)


Xs, Y = prepare_data(df)

logit = LogisticRegression(max_iter = 500)

logit.fit(Xs, Y)
pred_logit = logit.predict(Xs)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)
Xtest, Ytest = prepare_data(dtest)
pred_test = logit.predict(Xtest)

print(logit.intercept_)
print(logit.coef_)

from sklearn.feature_selection import RFECV

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=logit, step=1, cv=None, #StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
rfecv.fit(Xs, Y)

print("Optimal number of features : %d" % rfecv.n_features_)

predictors = rfecv.support_
for i in range(0, len(predictors)):
    if not predictors[i]:
        print(Xs.columns[i])

print(accuracy_score(Y, pred_logit))
print(confusion_matrix(Y, pred_logit))

print(accuracy_score(Ytest, pred_test))
print(confusion_matrix(Ytest, pred_test))

print('###################')

Ynb = rfecv.predict(Xs)
print(accuracy_score(Y, Ynb))
print(confusion_matrix(Y, Ynb))
Ynbtest = rfecv.predict(Xtest)
print(accuracy_score(Ytest, Ynbtest))
print(confusion_matrix(Ytest, Ynbtest))
