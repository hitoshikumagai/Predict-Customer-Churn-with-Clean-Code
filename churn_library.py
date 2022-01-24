# library doc string
"""
Library for Churn Prediction
Author Hitoshi Kumagai
Date January 2022
"""
#import libraries
#from typing_extensions import Self
#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(data_pth):
    '''returns dataframe for the csv found at data path
    input:
        data_pth: a path to the csv
    output:
        churn_data: pandas dataframe
    '''
    churn_data = pd.read_csv(data_pth)
    churn_data = churn_data.loc[:, churn_data.columns.values[2:]]
    return churn_data


def perform_eda(churn_data):
    '''
    perform eda on churn_data and save figures to images folder
    input:
        churn_data: pandas dataframe

    output:
        None
    '''
    churn_data['Churn'] = churn_data['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)
    print('Add Churn column for label column')

    directory_pth = './images/'

    hist_feature_lst = ['Churn', 'Customer_Age']
    for hist_feature in hist_feature_lst:
        fig = plt.figure(figsize=(20, 10))
        sns.histplot(churn_data[hist_feature])
        fig.savefig(directory_pth + hist_feature + '.png')
        print(f'Store {hist_feature} figure in {directory_pth}')

    bar_feature_lst = ['Marital_Status']
    for bar_feature in bar_feature_lst:
        fig = plt.figure(figsize=(20, 10))
        churn_data[bar_feature].value_counts('nomarize').plot(kind='bar')
        fig.savefig(directory_pth + bar_feature + '.png')
        plt.close()
        print(f'Store {bar_feature}  figure in {directory_pth}')

    dist_feature_lst = ['Total_Trans_Ct']
    for dist_feature in dist_feature_lst:
        fig = plt.figure(figsize=(20, 10))
        sns.distplot(churn_data[dist_feature])
        fig.savefig(directory_pth + bar_feature + '.png')
        plt.close()
        print(f'Store {dist_feature}  figure in {directory_pth}')

    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(churn_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig('./images/heatmap.png')
    plt.close()
    print('Store heatmap in ./images/heatmap/')


def encoder_helper(churn_data, categorical_columns, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        churn_data: pandas dataframe
        categorivcal_columns: list of columns that contain categorical features
        response: string of response name [optional argument that
                could be used for naming variables or index y column]
    output:
        df: pandas dataframe with new columns for
    '''
    churn_data['Churn'] = churn_data[response]\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)
    print('Add Churn column for label column')

    for category_column in categorical_columns:
        process_lst = []
        process_groups = churn_data.groupby(
            category_column).mean()['Churn']

        for val in churn_data[category_column]:
            process_lst.append(process_groups.loc[val])
        print(f'{category_column} column is transformed into quantative value.')

        churn_data[category_column + '_' + 'Churn'] = process_lst
        churn_data = churn_data.drop(category_column, axis=1)
        print(
            f"Add {category_column}_Churn column and drop {category_column} column")

    churn_data = churn_data.drop(response + '_Churn', axis=1)
    print('Drop Attrition_Flag_Churn column that are not used for prediction')
    return churn_data


def perform_feature_engineering(churn_data, response):
    '''
    Set up the Feature and Label columns from
    the data frame and split each column into training data and test data.

    input:
        churn_data: pandas dataframe
        response: string of response name [optional argument that
                could be used for naming variables or index y column]
    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    y_output = churn_data.loc[:, response]
    x_output = churn_data.drop(response, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_output, y_output, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test, output_pth):
    '''
    Train the prediction model using the training data, and store
    the prediction model in the specified folder.
    Also store the ROC curve images obtained from
    the test data in the specified folder.
    input:
        x_train: x training data
        x_test: x testing data
        y_train: y training data
        y_test: y testing data
    output:

    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    rfc_model = cv_rfc.best_estimator_
    print('Predict Random Forest Classifier by Grid Search')

    lrc = LogisticRegression()
    lrc.fit(x_train, y_train)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    print('Predict Logistic Regression')

    # Save model
    joblib.dump(rfc_model, output_pth +
                str(cv_rfc.best_estimator_).split('(')[0] + '.pkl')
    print('Random Forest Classifier model is saved')
    joblib.dump(lrc, output_pth + str(lrc).split('(')[0] + '.pkl')
    print('Logistic Regression model is saved')

    # Create ROC curve then save image
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    print('Create ROC curve for Logistic Regression and Random Forest Classifier')
    fig.savefig(output_pth + 'ROC_curve.png')
    plt.close()
    print(f'ROC curve image is stored in {output_pth}')

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
        output_pth: folder path of image folder

    output:
             None
    '''
    # Result of Random Forest classification
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.savefig(output_pth + 'Random_Forest_result.png')
    print('Save Random Forest report figure')

    # Result of Logistic Regression
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(output_pth + 'Logistic_Regression_result.png')
    print('Save Logistic Regression report figure')


def feature_importance_plot(model, x_data, output_pth):
    '''
    Create a graph of feature importances and store the result
    as an image in the folder specified by pth.
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            output_pth: path to store the figure
    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [x_data.columns[i] for i in indices]

    # Create bar plot for feature importances
    fig = plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    fig.savefig(
        output_pth +
        str(model).split('(')[0] +
        '_matplotlib_feature_importance.png')
    print('Save figure of feature importance related to Random Forest by matplotlib')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(
        output_pth +
        str(model).split('(')[0] +
        '_shap_feature_importance.png')
    print('Save figure of feature importance related to Random Forest by shap')


if __name__ == '__main__':
    churn_df = import_data("./data/bank_data.csv")
    perform_eda(churn_df)
    column_lst = churn_df.columns.values
    numeric_lst = churn_df.describe().columns.values
    category_lst = list(set(column_lst) - set(numeric_lst))
    churn_df = encoder_helper(churn_df, category_lst, 'Attrition_Flag')
    X_train, X_test, Y_train, Y_test = \
        perform_feature_engineering(churn_df, 'Churn')
    y_preds_lr_train, y_preds_rf_train, y_preds_lr_test, y_preds_rf_test =\
        train_models(X_train, X_test, Y_train, Y_test, "./models/")
    classification_report_image(
        Y_train,
        Y_test,
        y_preds_lr_train,
        y_preds_rf_train,
        y_preds_lr_test,
        y_preds_rf_test,
        "./images/")
    rfc_model_load = joblib.load('./models/RandomForestClassifier.pkl')
    feature_importance_plot(rfc_model_load, X_train, "./models/")
