import pandas as pd

from math import floor

__TRAIN_RATIO = 0.6
__VALID_RATIO = 0.2
__TEST_RATIO = 0.2


# Returns processed_train, processed_test
def preprocessing_data(data):
    targets = data.iloc[:, -1]
    data_no_targets = data.iloc[:, :-1]

    processed_data_no_targets = data_no_targets.fillna(data_no_targets[:data_no_targets.shape[0]].mean())
    processed_data_no_targets = pd.get_dummies(processed_data_no_targets)

    processed_data = pd.concat([processed_data_no_targets, targets], axis=1)

    num_elem = processed_data.shape[0]

    processed_train_data = processed_data[:floor(__TRAIN_RATIO * num_elem)]
    processed_valid_data = processed_data[
                           floor(__TRAIN_RATIO * num_elem):floor((__TRAIN_RATIO + __VALID_RATIO) * num_elem)]
    processed_test_data = processed_data[floor((__TRAIN_RATIO + __VALID_RATIO) * num_elem):]

    return processed_train_data, processed_valid_data, processed_test_data
