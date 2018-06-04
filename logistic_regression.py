#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 2nd June, 2018
Description: Logistic Regression module to do binary classification using sigmoid function
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utilities.util_functions import sigmoid
from dataset.dataset import logistic_regression


class LogisticRegression:

    def __init__(self, x, y):
        """
        Initialize the model with training data
        :param x: Numpy array of training input
        :param y: Numpy array of labelled output

        Size of matrix x: number of training examples * number of features of training examples
        Size of matrix y: number of training examples * 1
        """

        # Splitting the x and y matrix into train and test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=20)
        self.m, self.n = x.shape

        # Initialize the weights and biases
        self.w = np.random.randn(self.n, 1)
        self.c = np.random.randn(1, 1)

        # Set cost instance variable to None and cost_array instance to empty list
        self.cost = None
        self.cost_array = []

    @staticmethod
    def construct_training_matrices():
        """
        Constructs the training matrices of x and y here.  The values in x are given
        the square of the coordinates of the point inside and outside the circle.
        In y matrix, 1 refers that the point is inside the circle and 0 refers that the
        point is outside the circle.
        :return: x and y training matrices
        """

        # Create the training set using logistic_regression function of dataset module
        # x_i_j is ith feature when j is inside circle(1) or outside circle(0)
        x_1_1, x_2_1, x_1_0, x_2_0 = logistic_regression()

        # Show scatter plot of the training data with different marks representing
        # different labels using plot_example_data staticmethod
        LogisticRegression.plot_example_data(x_1_1, x_2_1, x_1_0, x_2_0)
        print("The dataset is plotted as shown in the screen.  Solid circle points \n"
              "and plus-tick points are datasets of different labels.\n")
        plt.show()

        # Concatenate the inside circle and outside circle points in a single array
        x_1 = np.concatenate((x_1_1, x_1_0))
        x_2 = np.concatenate((x_2_1, x_2_0))

        # Make whole training data into one input matrix, x and one output matrix, y
        x = np.array(list(zip(x_1, x_2))) ** 2
        y = np.concatenate((np.array([1] * 200), np.array([0] * 200))).reshape(-1, 1)

        return x, y

    @staticmethod
    def plot_example_data(x_1_1, x_2_1, x_1_0, x_2_0):

        # Show scatter plot of the training data with different marks representing
        # different labels
        plt.figure(figsize=(5, 10))
        plt.scatter(x_1_0, x_2_0, marker='+')
        plt.scatter(x_1_1, x_2_1, marker='o')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(-3.5, 3.5)
        plt.ylim(-6, 6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(('Outside circle', 'Inside circle'))

    def train(self, lambd, alpha):
        """
        One iteration of training and finding cost
        :param lambd: regularization parameter
        :param alpha: learning rate
        """

        a = sigmoid(np.matmul(self.x_train, self.w) + self.c)
        self.w = self.w - alpha * (np.matmul(self.x_train.T, a - self.y_train) / self.m + lambd * self.w / self.m)
        self.c = self.c - alpha * np.sum(a - self.y_train) / self.m

        self.cost = np.sum(- self.y_train * np.log(a + 10**(-10)) -
                           (1 - self.y_train) * np.log(1 - a + 10**(-10)))/self.m

    def fit(self, alpha, no_of_iterations, lambd):
        """
        Fits the training data to the linear model.
        :param alpha: learning rate
        :param no_of_iterations: no of iterations of gradient descent algorithm
        :param lambd: regularization parameter
        """

        # Training process to the given number of iterations
        print("Training the model...\n")
        for i in range(no_of_iterations):
            self.train(lambd, alpha)

            # Display cost in the epochs multiple of 10 and save cost in cost_array
            if i % 10000 == 0:
                self.cost_array.append(self.cost)
                print("Epoch {} ==> Cost: {}".format(i, self.cost))

    def predict(self, x):
        """
        Predicts the value of y on the basis of given value of x.
        :param x: input value of x, independent variable
        :return: value of predicted or dependent variable
        """
        return sigmoid(np.matmul(x, self.w) + self.c)

    def construct_boundary(self):
        """Construct the boundary"""

        # Create the training set using logistic_regression function of dataset module
        # x_i_j is ith feature when j is inside circle(1) or outside circle(0)
        x_1_1, x_2_1, x_1_0, x_2_0 = logistic_regression()

        # Show scatter plot of the training data with different marks representing
        # different labels using plot_example_data staticmethod
        LogisticRegression.plot_example_data(x_1_1, x_2_1, x_1_0, x_2_0)

        # Build the boundary using the weights and biases got by training
        # Upper and lower means above and below x-axis respectively
        x = np.linspace(-3, 3, 100)

        y_upper = np.sqrt(np.absolute((- self.c[0][0] - (x ** 2) * self.w[0][0]) / self.w[1][0]))
        y_lower = - np.sqrt(np.absolute((- self.c[0][0] - (x ** 2) * self.w[0][0]) / self.w[1][0]))
        plt.plot(x, y_upper, color='magenta')
        plt.plot(x, y_lower, color='magenta')
        plt.show()

    def validate(self):
        """
        Evaluates the performance the model by calculating and printing training
        and test accuracies and cross entropy function
        """
        # Predict test and train y's
        y_train_pred = self.predict(self.x_train) > 0.5
        y_test_pred = self.predict(self.x_test) > 0.5

        # Calculated train and test accuracy and print them
        train_accuracy = 100 * sum(y_train_pred == self.y_train) / self.y_train.shape[0]
        test_accuracy = 100 * sum(y_test_pred == self.y_test) / self.y_test.shape[0]

        print("\nAnd, the accuracy obtained is:")
        print("Training accuracy: {}%".format(train_accuracy))
        print("Test accuracy: {}%".format(test_accuracy))

    def graph_cost_vs_epochs(self):
        """Show graph of cost with respect to the epochs"""

        print("\nThe graph of cost vs epochs is as shown in the figure.")
        plt.plot(self.cost_array)
        plt.xlabel("No of epochs")
        plt.ylabel("Cross entropy cost")
        plt.show()


def main():

    # Use construct_training_matrices static method of LogisticRegression class
    # to construct and get the training data
    print("Constructing sample dataset of points lying inside circle and outside circle...")
    x, y = getattr(LogisticRegression, 'construct_training_matrices')()

    # Train the model and predict
    logistic_regressor = LogisticRegression(x, y)
    logistic_regressor.fit(0.1, 100000, 0)
    print("\nThe model has been trained.\nConstructing the boundary now ...\n")
    logistic_regressor.construct_boundary()
    print("The boundary is shown separating the two regions.\n")

    # Print weights and biases and the plot and also print the performance estimators of the model
    print("So, the weights and biases become:\nWeights:\n {}\nBiases:\n {}"
          .format(logistic_regressor.w, logistic_regressor.c))

    # Print the different accuracies and final cross entropy cost
    logistic_regressor.validate()

    # Draw graph of cost vs epochs
    logistic_regressor.graph_cost_vs_epochs()


if __name__ == "__main__":
    main()
