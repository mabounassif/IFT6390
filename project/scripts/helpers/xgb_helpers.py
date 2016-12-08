import pandas as pd

from math import floor

__TRAIN_RATIO = 0.6
__VALID_RATIO = 0.2
__TEST_RATIO = 0.2

__INFLUENTIAL_COLUMNS = ['OverallQual', 'Neighborhood', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea',
                         '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
                         'BsmtFinSF1']


# Returns processed_train, processed_test
def preprocess_data_and_separate(data):
    processed_data = preprocess_data(data)

    num_elem = processed_data.shape[0]

    processed_train_data = processed_data[:floor(__TRAIN_RATIO * num_elem)]
    processed_valid_data = processed_data[
                           floor(__TRAIN_RATIO * num_elem):floor((__TRAIN_RATIO + __VALID_RATIO) * num_elem)]
    processed_test_data = processed_data[floor((__TRAIN_RATIO + __VALID_RATIO) * num_elem):]

    return processed_train_data, processed_valid_data, processed_test_data


def preprocess_data(data):
    targets = data.iloc[:, -1]
    data_no_targets = data.iloc[:, :-1]
    processed_data_no_targets = data_no_targets.fillna(data_no_targets[:data_no_targets.shape[0]].mean())
    processed_data_no_targets = pd.get_dummies(processed_data_no_targets)
    processed_data = pd.concat([processed_data_no_targets, targets], axis=1)

    return processed_data


def prune_out_poor_correlation_columns_train(data):
    train_columns = __INFLUENTIAL_COLUMNS.copy()
    train_columns.append('SalePrice')

    return data[train_columns]


def prune_out_poor_correlation_columns_test(data):
    train_columns = __INFLUENTIAL_COLUMNS

    return data[train_columns]
