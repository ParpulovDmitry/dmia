import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import logging

shift = 200
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)

def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


train = pd.read_csv("../train_big.csv")
test = pd.read_csv("../test_big.csv")


ids = pd.read_csv('../data/test.csv')['id']

ids_train = train["id"]
ids_test = test["id"]

train_y = np.log(train['loss'] + shift)
train_x = train.drop(['loss','id'], axis=1)
test_x = test.drop(['loss','id'], axis=1)

n_folds = 3
cv_sum = 0
fpred = []
xgb_rounds = []

d_train_full = xgb.DMatrix(train_x, label=train_y)
d_test = xgb.DMatrix(test_x)

kf = KFold(train_x.shape[0], n_folds = n_folds, shuffle = True, random_state = 42)

oof_df = pd.DataFrame()

rand_state = 2016

params = {
          'seed': rand_state,
          'colsample_bytree': 0.7,
          'silent': 1,
          'subsample': 0.8,
          'learning_rate': 0.003,
          'objective': 'reg:linear',
          'max_depth': 6,
          'min_child_weight': 100,
          'booster': 'gbtree',
          'nthreads': 24
        }

for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d' % (i+1))
    X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
    y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]
    ind_tr, ind_val = ids_train.loc[train_index].values, ids_train.loc[test_index].values
    
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

    clf = xgb.train(params,
                    d_train,
                    500000,
                    watchlist,
                    early_stopping_rounds=150,
                    obj=fair_obj,
                    feval=xg_eval_mae,
                    verbose_eval = 100
                    )

    xgb_rounds.append(clf.best_iteration)
    scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    
    oof_df = oof_df.append(pd.DataFrame(np.vstack([ind_val, y_val, scores_val]).T, 
                               columns=["id", "y_true", "y_pred"]).set_index("id"))

    cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
    print('eval-MAE: %.6f' % cv_score)
    ### predict for test
    y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    
    pred = fpred
    cv_sum = cv_sum + cv_score


oof_df = oof_df.sort_index()    
oof_df.to_csv("oof_df.csv")


mpred = pred / n_folds   # average predictions
score = cv_sum / n_folds
print('Average eval-MAE: %.6f' % score)
n_rounds = int(np.mean(xgb_rounds))

print("Writing results")
result = pd.DataFrame(mpred, columns=['loss'])
result["id"] = ids
result = result.set_index("id")
print("%d-fold average prediction:" % n_folds)

now = datetime.now()
score = str(round((cv_sum / n_folds), 6))
sub_file = 'submission_10fold-average-xgb_fairobj_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("Writing submission: %s" % sub_file)
result.to_csv(sub_file, index=True, index_label='id')
"""
print 'All set predictions'
fpred = []
#################### ALL SET  ##########################
for rnd_seed in range(10,20):

    print 'seed: %s' % rnd_seed

    d_train = xgb.DMatrix(train_x, label=train_y)
    
    params = {
          'seed': rnd_seed,
          'colsample_bytree': 0.7,
          'silent': 1,
          'subsample': 0.8,
          'learning_rate': 0.003,
          'objective': 'reg:linear',
          'max_depth': 6,
          'min_child_weight': 100,
          'booster': 'gbtree',
          'nthreads': 24
        }


    clf = xgb.train(params,
                    d_train,
                    500000,
                    early_stopping_rounds=150,
                    #obj=fair_obj,
                    #feval=xg_eval_mae
                    )

    y_pred_all = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift
    fpred.append(y_pred_all)

pred = np.array(fpred).mean(axis=0)

print("Writing results")
result_all = pd.DataFrame(pred, columns=['loss'])
result_all["id"] = ids
result_all = result_all.set_index("id")
print("10 seed average prediction:")

now = datetime.now()
sub_file = 'submission_10_SEED-average-xgb_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("Writing submission: %s" % sub_file)
result_all.to_csv(sub_file, index=True, index_label='id')
"""
