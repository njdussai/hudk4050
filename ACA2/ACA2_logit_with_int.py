import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
df = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)

df.rename(columns = {   'Total Time' : 'Total_time', 
                        'totalobs-forsession' : 'totalobs_forsession',
                        'Obsv/act' : 'Obsv_act',
                        'Transitions/Durations' : 'Transitions_Durations'}, inplace = True)

cat_vars = ['SCHOOL', 'GRADE', 'CODER', 'Activity', 'Gender']
cont_vars = ['OBSNUM', 'totalobs_forsession','TRANSITIONS','FORMATchanges', 'Obsv_act', 'Transitions_Durations', 'Total_time']

patsy_str = 'ONTASK ~ ('

for c in cat_vars:
    patsy_str += 'C(' + c + ')' + (' * ' if c != 'Gender' else ') / (')

for c in cont_vars:
    patsy_str += '(' + c + ' - np.mean(' + c + '))' + (' + ' if c != 'Total_time' else ')')

ontask, X = dmatrices(patsy_str, data=df, return_type='dataframe')
y = ontask ['ONTASK[Y]']

### Regression ###
logit = LogisticRegression(max_iter = 10000, tol=1e-4)


logit.fit(X, y)
pred_logit = logit.predict(X)


dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)

dtest.rename(columns = {   'Total Time' : 'Total_time', 
                        'totalobs-forsession' : 'totalobs_forsession',
                        'Obsv/act' : 'Obsv_act',
                        'Transitions/Durations' : 'Transitions_Durations'}, inplace = True)

ontasktest, Xtest = dmatrices(patsy_str, data=dtest, return_type='dataframe')
ytest = ontasktest['ONTASK[Y]']

pred_test = logit.predict(Xtest)

print(accuracy_score(y, pred_logit))
print(confusion_matrix(y, pred_logit))

print(accuracy_score(ytest, pred_test))
print(confusion_matrix(ytest, pred_test))