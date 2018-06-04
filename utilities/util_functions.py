"""
Module containing utilities functions like finding sigmoid of the matrix or vector or value.
"""

import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


