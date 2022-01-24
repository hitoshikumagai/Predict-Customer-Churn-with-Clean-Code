"""
    Churn project test and log function
    Author: Hitoshi Kumagai
    Date: January 2022
"""

#from lib2to3.pgen2.pgen import DFAState
import os
import logging
import glob
import churn_library as cls

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import:Test import_data with logging
    input
        import_data:returns dataframe for the csv found at data path
    output
        churn_data: pandas dataframe extracting data from csv file
    '''
    try:
        churn_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

        assert churn_data.shape[0] > 0
        assert churn_data.shape[1] > 0
        col_number = churn_data.shape[0]
        row_number = churn_data.shape[1]
        logging.info("Datafarame column number is \
            {0:d} and row number is {1:d}: SUCCESS".format(col_number, row_number))
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found :FAIL")
        raise err
    return churn_data


def test_eda(perform_eda, churn_data):
    '''
        test perform eda function with logging
        input
            perform_eda :perform eda on churn_data and save figures to images folder
            churn_data : churn dataframe to perform eda function
        output
            None
    '''
    try:
        perform_eda(churn_data)
        directory_path = './images/'
        dir_val = os.listdir(directory_path)
        assert len(dir_val) == 5
        logging.info(
            "Testing perform_eda:Images are stored correctry :SUCCESS")
    except AssertionError as err:
        logging.warning(
            "Testing perform_eda Images are not save to the eda folder.:FAIL")
        raise err


def test_encoder_helper(encoder_helper, churn_data, category_lst_, response):
    '''
    test encoder helper with logging
    input
        encoder_helper:helper function to turn each categorical column into a new column with
            propotion of churn for each category - associated with cell 15 from the notebook
        churn_data:pandas dataframe
        category_lst_:list of columns that contain categorical features
        response:string of response name [optional argument that
        could be used for naming variables or index y column]
    output
        churn_data:pandas dataframe
    '''
    churn_data = encoder_helper(churn_data, category_lst_, response)

    try:
        churn_count = 0
        for col in churn_data.columns.values:
            if '_Churn' in col:
                churn_count += 1
        assert churn_count == 5
        logging.info(
            'the {churn_count} Categorical column are transformed:SUCCESS')
    except AssertionError as err:
        logging.warning(
            "The categorical columns are not specified correctly:FAIL")
        raise err
    return churn_data


def test_perform_feature_engineering(
        perform_feature_engineering,
        churn_data,
        response):
    '''
    test perform_feature_engineering with logging
    input
        perform_feature_engineering:
            Set up the Feature and Label columns from
            the data frame and split each column into training data and test data.
        churn_data:Encoded pandas dataframe
        response:Label column for prediction
    output
        x_train:X training data
        x_test:X testing data
        y_train:y training data
        y_test:y testing data
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            churn_data, response)
        assert x_train.shape[0] == 7088
        assert x_train.shape[1] == 19
        assert x_test.shape[0] == 3039
        assert x_test.shape[1] == 19
        assert y_train.shape[0] == 7088
        assert y_test.shape[0] == 3039
        logging.info(
            "The input data was correctly split into training data and test data.:SUCCESS")
    except AssertionError as err:
        logging.warning(" :FAIL")
        raise err
    return x_train, x_test, y_train, y_test


def test_train_models(
        train_models,
        x_train,
        x_test,
        y_train,
        y_test,
        response):
    '''
    test train_models
        Train the prediction model using the training data, and store
        the prediction model in the specified folder.
        Also store the ROC curve images obtained from
        the test data in the specified folder.
    input
        train_models:
        x_train:X training data
        x_test:X testing data
        y_train:y training data
        y_test:y testing data
        response:Label column for prediction
    output
        None
    '''
    try:
        train_models(x_train, x_test, y_train, y_test, response)
        model_val = glob.glob(response + '*.pkl')
        assert len(model_val) == 2
        logging.info("Models are saved correctly in target folder:SUCSESS")
        image_val = glob.glob(response + '*.png')
        assert len(image_val) == 1
        logging.info("Image is saved correctly in target folder:SUCSESS")
    except AssertionError as err:
        logging.warning("Models and image is not saved correctly  :FAIL")
        raise err


if __name__ == "__main__":
    data_frame = test_import(cls.import_data)
    test_eda(cls.perform_eda, data_frame)
    column_lst = data_frame.columns.values
    numeric_lst = data_frame.describe().columns.values
    category_lst = list(set(column_lst) - set(numeric_lst))
    data_frame = test_encoder_helper(
        cls.encoder_helper,
        data_frame,
        category_lst,
        'Attrition_Flag')
    x_train_, x_test_, y_train_, y_test_ = test_perform_feature_engineering(
        cls.perform_feature_engineering, data_frame, 'Churn')
    test_train_models(
        cls.train_models,
        x_train_,
        x_test_,
        y_train_,
        y_test_,
        "./models/")
