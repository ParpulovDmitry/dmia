import pandas as pd
import numpy as np

import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error 
from rgf.rgf import RGFRegressor


train = pd.read_csv("train_big.csv")
test = pd.read_csv("test_big.csv")

shift = 200
ids = pd.read_csv('data/test.csv')['id']
y_train = np.log(train['loss'] + shift)
X_train = train.drop(['loss','id'], axis=1)
X_test = test.drop(['loss','id'], axis=1)

rgf = RGFRegressor(max_leaf=50000,
                   algorithm="RGF",
                   test_interval=500, loss='LS', l2=0.01, sl2=None, reg_depth=1, verbose=0)

rgf.fit(X_train, y_train)

pred_rgf = np.exp(rgf.predict(X_test)) - shift

with open('pred_rgf.pickle', 'wb') as f:
    pickle.dump(pred_rgf, f)

test.loc[:, 'loss'] = pred_rgf
test.reset_index()[['id', 'loss']].to_csv('submissions/rgf_50000_0.01.csv', index = False)

