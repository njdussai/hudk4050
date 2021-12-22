import numpy as np
import pandas as pd


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

Xtrain = Xtrain.to_numpy()
Ytrain = Ytrain.to_numpy()

#### Keras

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=26, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, epochs=10, batch_size=50)

_, accuracy = model.evaluate(Xtrain, Ytrain)
print('Accuracy: %.2f' % (accuracy*100))