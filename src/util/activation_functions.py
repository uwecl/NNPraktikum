# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np
from math import *


class Activation:
    """
    Containing various activation functions
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput < threshold

    @staticmethod
    def sigmoid(netOutput):
        return 1/(1+e**(-1.0*netOutput))

    @staticmethod
    def tanh(netOutput):
        pass

    @staticmethod
    def rectified(netOutput):
        pass

    @staticmethod
    def softmax(netOutput):
        pass
