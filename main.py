#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression


def get_train_matrices():

    # Read the csv files
    df_x = pd.read_csv('train/input.csv')
    df_y = pd.read_csv('train/output.csv')

    # Make training data as numpy arrays
    x = df_x.values[:, 1:]
    y = df_y.values[:, 1:]

    return x, y


def main():

    # Get training matrices for linear regression model
    x, y = get_train_matrices()

    # Create instance of LinearRegression with the training matrices
    linear_regression = LinearRegression(x, y)

    # Fit with learning rate, no of iterations and regularization(L2) parameter
    linear_regression.fit(0.01, 1000, 0)

    # Predict for all the input values
    y_pred = linear_regression.predict(x)

    # Plot the scatter plots of training data and graph of our linear model
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()

    # Print the weights and biases of the model
    print("Weights: {}\nBiases: {}".format(linear_regression.w, linear_regression.c))

    # Validate the model by printing the performance metrics
    linear_regression.validate()

    # Predict for the input data in test folder and save as output.csv in test folder
    x_test = pd.read_csv('test/input.csv')['x'].values.reshape(-1,1)
    y_test = linear_regression.predict(x_test)
    df_predict = pd.DataFrame({'y': y_test.reshape(-1)})
    df_predict.to_csv('test/output.csv')


if __name__ == "__main__":
    main()
