import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt

from project.scripts.helpers.xgb_helpers import preprocess_data, prune_out_poor_correlation_columns_train, \
    prune_out_poor_correlation_columns_test

np.random.seed(45432)

data = pd.read_csv('../data/train.csv')
kaggle_test = pd.read_csv('../data/test.csv')

shuffled_data = data.reindex(np.random.permutation(data.index))

processed_data = prune_out_poor_correlation_columns_train(shuffled_data)
processed_data = preprocess_data(processed_data)

processed_kaggle_test = prune_out_poor_correlation_columns_test(kaggle_test)
processed_kaggle_test = preprocess_data(processed_kaggle_test)

xgb_data = processed_data.as_matrix()[:, :-1]
xgb_label = processed_data.as_matrix()[:, -1]

# load data in order to do training
dtrain = xgb.DMatrix(xgb_data, label=xgb_label)
param = {
    'colsample_bytree': 0.3,
    'colsample_bylevel': 0.2,
    'gamma': 1.0,
    'learning_rate': 0.5,
    'min_child_weight': 4,
    'lambda': 0.9,
    'subsample': 0.5,
    'seed': 42,
    'max_depth': 5,
    'silent': 1,
    'objective': 'reg:linear'
}

num_round = 50

print('running cross validation')
res = xgb.cv(param, dtrain, num_round, nfold=5,
             metrics={'rmse'}, seed=45432,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print(res)

booster = xgb.train(param, dtrain, num_round)
preds = booster.predict(xgb.DMatrix(processed_kaggle_test.as_matrix()))

plot = xgb.plot_tree(booster)
plt.plot(plot)

df_preds = pd.DataFrame(preds, index=kaggle_test["Id"], columns=["SalePrice"])
df_preds.to_csv('output.csv', header=True, index_label='Id')
