#!/usr/bin/env python3

import pandas as pd
from logistic_regression import LogisticRegression


def get_train_matrices():

    # Read the csv files
    df_x = pd.read_csv('train/input.csv')
    df_y = pd.read_csv('train/output.csv')

    # Make training data as numpy arrays
    x = df_x.values[:, 1:]
    y = df_y.values[:, 1:]

    return x, y


def main():

    # Get training matrices for logistic regression model
    x, y = get_train_matrices()

    # Create instance of LogisticRegression with the training matrices
    logistic_regression = LogisticRegression(x, y)

    # Fit with learning rate, no of iterations and regularization(L2) parameter
    logistic_regression.fit(0.01, 100000, 0)

    # Print weights and biases and the plot and also print the performance estimators of the model
    print("So, the weights and biases become:\nWeights:\n {}\nBiases:\n {}"
          .format(logistic_regression.w, logistic_regression.c))

    # Validate the model by printing the performance metrics
    logistic_regression.validate()

    # Graph the curve of cost vs no of epochs
    logistic_regression.graph_cost_vs_epochs()

    # Predict for the input data in test folder and save as output.csv in test folder
    x_test = pd.read_csv('test/input.csv').values[:, 1:]
    y_test = logistic_regression.predict(x_test)
    df_predict = pd.DataFrame({'y': y_test.reshape(-1)})
    df_predict.to_csv('test/output.csv')


if __name__ == "__main__":
    main()
