"""Summary
"""
import sys
sys.path.insert(0, '..')
from src.investsai.sai import SAI
import pytest
from typing import Union
import random
import numpy as np
import pandas as pd


random.seed(99)
np.random.seed(99)


def simulate_data() -> Union[dict, dict, dict]:
    """
    This function simulates data to demonstrate the module

    Returns:
        Union[dict, dict, dict]: simulate training and testing data
    """

    # Simulate n securities each with m numeric variables/factors
    n = 100  # simulate 1000 securities
    m = 5  # each securities with 20 variables/factors

    # a pandas DataFrame with training data n x m. n=number of securities and m=number of variables/factors

    XTrain = pd.DataFrame(np.random.rand(n, m - 1), columns=[
        f'factor_{i}' for i in range(m - 1)], index=['sec' + str(i) for i in range(1, n + 1)])
    # add a str factor (i.e. Sector) to test functionality, the dimension of input X is now n x m
    XTrain.loc[:, 'Sector'] = ['Technology'] * \
        int(n / 2) + ['Munufacturing'] * (n - int(n / 2))

    # Simulate a y varible to represent whether the stocks in training data is successful (can be by any objective or multi-objective)
    y = [0] * int(n * 0.70) + [1] * int(n - int(n * 0.70))
    random.shuffle(y)
    yTrain = pd.DataFrame(y, columns=['y'])

    # Simulate a test data with n_test securities each with the same m variables/factor
    n_test = 10000  # test on 10000 securities, can be different from n
    XTest = pd.DataFrame(np.random.rand(n_test, m - 1), columns=[f'factor_{i}' for i in range(m - 1)], index=[
        'sec' + str(i) for i in range(1, n_test + 1)])
    # added another variable to represent the sector, the dimension of input XTest is now n x m
    XTest.loc[:, 'Sector'] = ['Technology'] * \
        int(n_test / 2) + ['Munufacturing'] * (n_test - int(n_test / 2))

    return XTrain, yTrain, XTest


def test_parameters():
    """test model parameters. need to discretize vaariables more than 2 categories
    """
    params = {
        'q': 0,
        'parallel': True,
        'nb_workers': 2
    }

    with pytest.raises(ValueError):
        invest_sai = SAI(params=params)


def test_input_data_same_dim() -> None:
    """test input XTrian and yTrain need to have the same dimension
    """
    params = {
        'q': 3,
        'parallel': True,
        'nb_workers': 2
    }

    invest_sai = SAI(params=params)

    XTrain, yTrain, XTest = simulate_data()
    yTrain = yTrain.iloc[:-10]

    with pytest.raises(ValueError):
        invest_sai.fit(X=XTrain, y=yTrain)


def test_X_format() -> None:
    """test X format
    """
    XTrain, yTrain, XTest = simulate_data()
    params = {
        'q': 3,
        'parallel': True,
        'nb_workers': 2
    }
    XTrain = XTrain.values
    invest_sai = SAI(params=params)

    with pytest.raises(ValueError):
        invest_sai.fit(X=XTrain, y=yTrain)


def test_y_format() -> None:
    """test y format
    """
    XTrain, yTrain, XTest = simulate_data()
    params = {
        'q': 3,
        'parallel': True,
        'nb_workers': 2
    }
    yTrain = yTrain.values
    invest_sai = SAI(params=params)

    with pytest.raises(ValueError):
        invest_sai.fit(X=XTrain, y=yTrain)


def test_xTest_format() -> None:

    XTrain, yTrain, XTest = simulate_data()
    params = {
        'q': 3,
        'parallel': True,
        'nb_workers': 2
    }
    invest_sai = SAI(params=params)
    invest_sai.fit(X=XTrain, y=yTrain)
    XTest = XTest.values

    with pytest.raises(ValueError):
        invest_sai.predict(XTest)
