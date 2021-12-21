
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import pandas as pd
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

def prepare_data(df):
    dfx = df.loc[:, df.columns != 'ONTASK']
    Xs = pd.get_dummies(dfx, columns = ['SCHOOL', 'GRADE', 'CODER', 'Activity'])

    Y = df['ONTASK']#.replace(to_replace = ['Y', 'N'], value = [1, 0])#.to_numpy()
    return Xs, Y

#Import data
df = pd.read_csv('aca2_dataset/aca2_dataset_training.csv', usecols = cols)
Xs, Y = prepare_data(df)

dtest = pd.read_csv('aca2_dataset/aca2_dataset_validation.csv', usecols = cols)
Xtest, Ytest = prepare_data(dtest)

###############################

svc = SVC(kernel="linear", max_iter = 1000000)
# The "accuracy" scoring shows the proportion of correct classifications

svc.fit(Xs,Y)
Ysvc = svc.predict(Xs)
print(accuracy_score(Y, Ysvc))
print(confusion_matrix(Y, Ysvc))
Ysvctest = svc.predict(Xtest)
print(accuracy_score(Ytest, Ysvctest))
print(confusion_matrix(Ytest, Ysvctest))

quit()
min_features_to_select = 26  # Minimum number of features to consider
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
rfecv.fit(Xs, Y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()