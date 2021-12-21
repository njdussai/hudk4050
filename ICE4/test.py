import pandas as pd

mooc = pd.read_csv("ICE4_Data.csv")
dummy = pd.get_dummies(mooc['certified'], prefix = 'certified')
moocD = pd.concat([mooc, dummy], axis=1)
moocD = moocD.drop(['certified', 'certified_no'], axis=1)
'''

mooc = pd.read_csv("ICE4_Data.csv")
mooc.replace(to_replace=['yes', 'no'] , value=[1, 0], inplace=True)

Xs = mooc[["forum.posts", "grade", "assignment"]]#.to_numpy()
certified = mooc["certified"]#.to_numpy()

'''
Xs = moocD[["forum.posts", "grade", "assignment"]]
certified = moocD["certified_yes"].astype('int64')


from sklearn.model_selection import train_test_split

## For logistic regression
xs_logit_training, xs_logit_test, y_logit_training, y_logit_test = train_test_split(Xs, certified, test_size = 0.2, random_state=0)

import statsmodels.api as sm

xs_logit_training = sm.add_constant(xs_logit_training)#.astype('int64')

smlogit = sm.Logit( y_logit_training, xs_logit_training)

smlogitFit = smlogit.fit(method='ncg', maxiter = 200)#, avextol = 1e-20)

certified_pred_logit2 = smlogitFit.predict(sm.add_constant(xs_logit_test)) >= 0.5

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_logit_test, certified_pred_logit2))

#print(smlogitFit.summary())
#print(Xs)
#print(certified)