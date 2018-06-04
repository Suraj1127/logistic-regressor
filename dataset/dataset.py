"""
Module that prepares naive dataset for different machine learning models.
"""
import numpy as np


def logistic_regression():
    """
    Constructs the training matrices of x and y here for logistic regression.  The values in x
    are given the square of the coordinates of the point inside and outside the circle.
    In y matrix, 1 refers that the point is inside the circle and 0 refers that the
    point is outside the circle.

    :return:
    x_1_1: x_1 feature for points inside the circle
    x_2_1: x_2 feature for points inside the circle
    x_1_0: x_1 feature for points outside the circle
    x_2_0: x_2 feature for points outside the circle
    """

    # Points inside the circle
    # Points inside the circle(radius=3) above the x-axis
    x1_positive_1 = np.linspace(-2.99, 2.99, 100)
    x2_positive_1 = np.sqrt(9 - x1_positive_1 ** 2) * np.random.rand(100)

    # Points inside the circle(radius-3) below the x-axis
    x1_negative_1 = np.linspace(-2.99, 2.99, 100)
    x2_negative_1 = - np.sqrt(9 - x1_negative_1 ** 2) * np.random.rand(100)

    # Points outside the circle

    # Outside points above x-axis
    x1_positive_0 = np.linspace(-2.99, 2.99, 100)
    x2_positive_0 = np.sqrt(9 - x1_positive_1 ** 2) * (1 + np.random.rand(100))

    # Outside points below x-axis
    x1_negative_0 = np.linspace(-2.99, 2.99, 100)
    x2_negative_0 = - np.sqrt(9 - x1_positive_1 ** 2) * (1 + np.random.rand(100))

    # Concatenate all the above circles and below circles point(inside the circle)
    x_1_1 = np.concatenate((x1_positive_1, x1_negative_1))
    x_2_1 = np.concatenate((x2_positive_1, x2_negative_1))

    # Concatenate all the above circles and below circles point(outside the circle)
    x_1_0 = np.concatenate((x1_positive_0, x1_negative_0))
    x_2_0 = np.concatenate((x2_positive_0, x2_negative_0))

    return x_1_1, x_2_1, x_1_0, x_2_0
