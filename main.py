"""Summary
"""
from src.investsai.sai import SAI
from typing import Union
import random
import numpy as np
import pandas as pd
import time


def simulate_data() -> Union[dict, dict, dict]:
    """
    This function simulates data to demonstrate the module

    Returns:
        Union[dict, dict, dict]: simulate training and testing data
    """

    # Simulate n securities each with m numeric variables/factors
    n = 10000  # simulate 1000 securities
    m = 20  # each securities with 20 variables/factors

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


def main() -> None:
    """
    Run the invest-sai algorithm and print learned rules and predictions
    """

    XTrain, yTrain, XTest = simulate_data()  # Simulate data

    # Start using Invest-SAI algorithm
    # Input parameters
    params = {
        'q': 3,
        'parallel': True,
        'nb_workers': 2
    }

    invest_sai = SAI(params=params)
    invest_sai.fit(X=XTrain, y=yTrain)

    # This is the output expected success probilities of the securities in the test data
    yTest = invest_sai.predict(X=XTest)

    print(
        f'\n\nInterpretable rules with conditional probailities to rank securities\n{invest_sai.rules}\n')
    print(f'\nRanking of securities in XTest\n{yTest}\n')


if __name__ == '__main__':

    st_wall, st_cpu = time.time(), time.process_time()
    main()
    elapsed_time_wall, elapsed_time_cpu = time.time(
    ) - st_wall, time.process_time() - st_cpu

    print('\nExecution time:', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time_wall)))
    print('CPU Execution time:', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time_cpu)))
