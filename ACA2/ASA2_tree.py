import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE


############ Prepare data ##############

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
    Xs = pd.get_dummies(dfx, columns = ['SCHOOL', 'GRADE', 'CODER', 'Activity'])

    Y = df['ONTASK'].replace(to_replace = ['Y', 'N'], value = [1, 0])
    return Xs, Y

#Import data
dtrain = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)
Xtrain, Ytrain = prepare_data(dtrain)

dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)
Xtest, Ytest = prepare_data(dtest)


############## Tree #################


