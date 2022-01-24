# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Reducing the churn rate by identifying and later intervening with customers who are likely to churn is an important task for various companies. In this study, I will create a model that can better predict churning customers by performing multiple machine learning models. Since the data analysis and prediction models have already been created, I will develop a library and a logging function as well as a testing function for the module so that the prediction models can be used continuously.

The data set was obtained from the following link.<br>
[Dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code)

## Resources

glob<br>
sklearn 1.0.2<br>
shap 0.39.0<br>
joblib<br>
pandas 1.3.0<br>
numpy 1.20.3<br>
matplotlib 3.3.4<br>
seaborn 0.11.1<br>
pylint 2.7.4<br>
autopep8 1.5.6<br>

If the version is different, please update the library by referring to the following command.

`
conda update pandas
`

or
`
pip3 install numpy --upgrade
`
## Files

root<br>
|-churn_library.py
|-churn_script_logging_and_tests.py<br>
|-README.md<br>
|-data<br>
&emsp;|-bank_data.csv<br>
|-images<br>
|-logs<br>
&emsp;|-churn_library.log<br>
|-models<br>

## Running Files
### Usage

Please type below to run this script to get full result.

`
python churn_script_logging_and_tests.py
`

Please type below to run this script with logging.

`
python churn_script_logging_and_tests.py
`

The following commands were used when maintaining the library and log files.

```
pylint churn_library.py
autopep8 --in-place --aggressive --aggressive churn_library.py

pylint churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```
