# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
from math import *
from util.activation_functions import Activation


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        # Here you have to calculate the MeanSquareError
        # MSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        pass


class CrossEntropyError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """

    def calculateError(self, target, output):
        pass
