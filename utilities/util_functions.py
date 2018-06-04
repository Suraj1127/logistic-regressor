"""
Module containing utilities functions like finding sigmoid of the matrix or vector or value.
"""

import numpy as np


def sigmoid(z):
    """
    Implements sigmoid function
    :param z: array or matrix or a number
    :return: sigmoid of the array or matrix or a number applied elementwise
    """
    return 1/(1+np.exp(-z))


