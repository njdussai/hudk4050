
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

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

    Y = df['ONTASK']#.replace(to_replace = ['Y', 'N'], value = [1, 0])#.to_numpy()
    return Xs, Y

#Import data
df = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)
Xs, Y = prepare_data(df)
#Xs = df.loc[:, df.columns != 'ONTASK']
#Y = df['ONTASK']

dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)
Xtest, Ytest = prepare_data(dtest)

############
from sklearn.naive_bayes import BernoulliNB

nb = BernoulliNB()
nb.fit(Xs, Y)
Ynb = nb.predict(Xs)
print(accuracy_score(Y, Ynb))
print(confusion_matrix(Y, Ynb))
Ynbtest = nb.predict(Xtest)
print(accuracy_score(Ytest, Ynbtest))
print(confusion_matrix(Ytest, Ynbtest))

#######
nbrfe = BernoulliNB()

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=nbrfe, step=1, cv=None, #StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
rfecv.fit(Xs, Y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
'''
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
'''
print(Xs.columns)
print(rfecv.support_)

Ynb = nb.predict(Xs)
print(accuracy_score(Y, Ynb))
print(confusion_matrix(Y, Ynb))
Ynbtest = nb.predict(Xtest)
print(accuracy_score(Ytest, Ynbtest))
print(confusion_matrix(Ytest, Ynbtest))

print('###################')

Ynb = rfecv.predict(Xs)
print(accuracy_score(Y, Ynb))
print(confusion_matrix(Y, Ynb))
Ynbtest = rfecv.predict(Xtest)
print(accuracy_score(Ytest, Ynbtest))
print(confusion_matrix(Ytest, Ynbtest))