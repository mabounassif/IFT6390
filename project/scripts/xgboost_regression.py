import pandas as pd
import numpy as np
import xgboost as xgb

from project.scripts.helpers.xgb_helpers import preprocessing_data

np.random.seed(45432)

data = pd.read_csv('../data/train.csv')
kaggle_test = pd.read_csv('../data/test.csv')

shuffled_data = data.reindex(np.random.permutation(data.index))

processed_train, processed_valid, processed_test = preprocessing_data(shuffled_data)

X = processed_train.as_matrix()[:, :-1]
y = processed_train.as_matrix()[:, -1]

xgb_model = xgb.XGBRegressor().fit(X, y)
preds = xgb_model.predict(processed_test.as_matrix()[:, :-1])

print(preds)
