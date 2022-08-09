'''
Module for performing test and logging for churn_library.py

- test data import
- test EDA
- test data encoder
- test feature enginering
- test train model

Author: Mahesh Sinha
Date: 09 August 2022
'''

# Import library:
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: FAILED: File not found.")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: FAILED: The file doesn't have rows and columns")
        raise err
    return data_frame


def test_eda(perform_eda, data_frame):
    '''
    test perform eda function
    '''
    perform_eda(data_frame)
    eda_path = "./images/eda"

    # check for all eda images:
    try:
        dir_val = os.listdir(eda_path)
        assert dir_val.count('churn_distribution.png')
        assert dir_val.count('customer_age_distribution.png')
        assert dir_val.count('heatmap.png')
        assert dir_val.count('marital_status_distribution.png')
        assert dir_val.count('total_trans_Ct.png')
        logging.info('Testing perform_eda: SUCCESS')
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: FAILED : unable to execute the test_eda function")
        raise err


def test_encoder_helper(encoder_helper, data_frame):
    '''
    test encoder helper
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    data_frame = encoder_helper(data_frame, category_lst, 'Churn')

    try:
        for col in category_lst:
            assert col in data_frame.columns
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: FAILED, looks like few category columns are missing.')
        raise err
    return data_frame


def test_perform_feature_engineering(perform_feature_engineering, data_frame):
    '''
    test perform_feature_engineering
    '''
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, 'Churn')

    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info('Testing perform_feature_engineering: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: FAILED, Missing data')
    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    # train and save models
    train_models(x_train, x_test, y_train, y_test)

    classification_reports_path = './images/results'
    try:
        count_files = os.listdir(classification_reports_path)
        assert len(count_files) > 0
    except FileNotFoundError as err:
        logging.error(
            'Testing train_models: FAILED, looks like classification reports are missing')
        raise err

    models_path = './models'
    try:
        check_model_dir = os.listdir(models_path)
        assert len(check_model_dir) > 0
        logging.info('Testing train_models: SUCCESS')
    except FileNotFoundError as err:
        logging.error(
            'Testing train_models: FAILED, looks like models are not available')


if __name__ == "__main__":
    data = test_import(cls.import_data)
    test_eda(cls.perform_eda, data)
    data = test_encoder_helper(cls.encoder_helper, data)
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, data)
    test_train_models(cls.train_models, x_train, x_test, y_train, y_test)
