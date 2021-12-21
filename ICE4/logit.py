#!/Users/ndussaillant/opt/anaconda3/bin/python3

import numpy as np
import statsmodels.api as sms
import pandas as pd
from sklearn.linear_model import LogisticRegression

mooc = pd.read_csv("ICE4_Data.csv")
mooc.replace(to_replace=['yes', 'no'] , value=[1, 0], inplace=True)

Xs = mooc[["forum.posts", "grade", "assignment"]]#.to_numpy()
Y = mooc["certified"]#.to_numpy()

from sklearn.model_selection import train_test_split
xs_logit_training, xs_logit_test, y_logit_training, y_logit_test = train_test_split(Xs, Y, test_size = 0.2, random_state=0)

# Sklearn
sklogit = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'newton-cg', max_iter = 100, tol=1e-4)
sklogit.fit(Xs, Y)

print(sklogit.n_iter_, sklogit.intercept_, sklogit.coef_)

print('\n')

# Statsmodels
smslogit = sms.Logit(y_logit_training, xs_logit_training)
smslogitFit = smslogit.fit(method='ncg', full_output = 1, maxiter = 100, avextol = 1e-20)

# Results

print("SMS")
print(smslogitFit.summary())

print(Xs)
print(Y)

