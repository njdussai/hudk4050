
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

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

    ord_enc = OrdinalEncoder()
    Xs_cat = ord_enc.fit_transform(dfx[['SCHOOL', 'GRADE', 'CODER', 'Activity', 'Gender']])
    Xs_cont = dfx[['OBSNUM', 'totalobs-forsession', 'TRANSITIONS', 'FORMATchanges','Obsv/act', 'Transitions/Durations', 'Total Time']]

    Y_cat = df['ONTASK']
    Y_cont = df['ONTASK'].replace(to_replace = ['Y', 'N'], value = [1, 0])
    return Xs_cat, Xs_cont, Y_cat, Y_cont

#Import data
df = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)
Xs_cat, Xs_cont, Y_cat, Y_cont = prepare_data(df)

dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)
Xs_catTest, Xs_contTest, Y_catTest, Y_contTest = prepare_data(dtest)

############
from sklearn.naive_bayes import CategoricalNB

nb = CategoricalNB()
nb.fit(Xs_cat, Y_cat)

logit = LogisticRegression()
logit.fit(Xs_cont, Y_cont)

ProbYcat = nb.predict_proba(Xs_cat)
ProbYcont = logit.predict_proba(Xs_cont)

Y_pred = pd.DataFrame((1 - ProbYcat[:, 0] * ProbYcont[:, 0]) >= 0.5).replace(to_replace = [True, False], value = ['Y', 'N'])

print(accuracy_score(Y_cat, Y_pred))
print(confusion_matrix(Y_cat, Y_pred))


'''
Ynbtest = nb.predict(Xtest)
print(accuracy_score(Ytest, Ynbtest))
print(confusion_matrix(Ytest, Ynbtest))

quit()

max = 0
for i in range(1, Xs.shape[1]):
    rfe = RFE(estimator = MultinomialNB, n_features_to_select = i)
    rfe_model = rfe.fit(Xs, Y)

    Yrfe_train = rfe_model.predict(Xs)
    Yrfe_test = rfe_model.predict(Xtest)

    print('Predictors: ', i)
    if accuracy_score(Y, Yrfe_train) > max:
        print('**** New max ****')
        max = accuracy_score(Y, Yrfe_train)
    print('Training accuracy: ', accuracy_score(Y, Yrfe_train))
    print('Validation accuracy: ', accuracy_score(Ytest, Yrfe_test))

    xcols = []
    for i in range(0, len(Xs.columns)):
        if rfe.support_[i]:
            xcols.append(Xs.columns[i])
    print(xcols)
    print('\n\n')
'''