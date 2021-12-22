import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFE
import pickle

def save_data(X, y, name):
    with open(name + '.pkl', 'wb') as outp:
        pickle.dump(X, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y, outp, pickle.HIGHEST_PROTOCOL)

def prepare_data(url, name):
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

    #Import data
    df = pd.read_csv(url, usecols = cols)

    df.rename(columns = {   'Total Time' : 'Total_time', 
                            'totalobs-forsession' : 'totalobs_forsession',
                            'Obsv/act' : 'Obsv_act',
                            'Transitions/Durations' : 'Transitions_Durations'}, inplace = True)

    cat_vars = ['SCHOOL', 'GRADE', 'CODER', 'Activity', 'Gender']
    cont_vars = ['OBSNUM', 'totalobs_forsession','TRANSITIONS','FORMATchanges', 'Obsv_act', 'Transitions_Durations', 'Total_time']
    #cont_vars = ['TRANSITIONS','FORMATchanges', 'Transitions_Durations']

    patsy_str = 'ONTASK ~ ('

    for i, c in enumerate(cat_vars):
        patsy_str += 'C(' + c + ')' + (' * ' if i != (len(cat_vars) - 1) else ') / (')

    for i, c in enumerate(cont_vars):
        patsy_str += c + (' + ' if i != (len(cont_vars) - 1) else ')')

    ontask, data = dmatrices(patsy_str, data=df, return_type='dataframe')
    scaler = StandardScaler()
    scaler.fit(data)
    X = scaler.transform(data)

    y = ontask ['ONTASK[Y]']

    save_data(X, y, name)

    return X, y

def retrieve_data(url, name):
    try:
        with open(name + ".pkl", "rb") as inp:
            X = pickle.load(inp)
            y = pickle.load(inp)
    except (OSError, IOError) as e:
        X, y = prepare_data(url, name)
    return X, y


X, y = retrieve_data('aca2_dataset/aca2_dataset_training.csv', 'training')
Xtest, ytest = retrieve_data('aca2_dataset/aca2_dataset_validation.csv', 'validation')

print('Start Regression')
### Regression ###
logit = LogisticRegression(max_iter = 10000, tol=1e-4, solver = 'sag')
logit.fit(X, y)
print('Number of iterations: ' + str(logit.n_iter_))
print('End Regression')

pred_logit = logit.predict(X)

pred_test = logit.predict(Xtest)

'''
logit = LogisticRegression(max_iter = 10000, tol=1e-2, solver = 'sag')
rfe = RFE(logit, n_features_to_select=1)
rfe.fit(X, y)
pred_logit = rfe.predict(X)
pred_test = rfe.predict(Xtest)
'''

print(accuracy_score(y, pred_logit))
print(confusion_matrix(y, pred_logit))

print(accuracy_score(ytest, pred_test))
print(confusion_matrix(ytest, pred_test))