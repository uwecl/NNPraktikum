# -*- coding: utf-8 -*-

import numpy as np
import logging

from util.activation_functions import Activation
from model.classifier import Classifier

__author__ = "Benjamin Rupp"
__email__ = "uwecl@student.kit.edu"


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self):
        """Train the Logistic Regression"""

        from util.loss_functions import DifferentError
        loss = DifferentError()
        
        learned = False
        iteration = 0

        #for epoch in range(self.epochs):
        while not learned:
            pos = 0
            grad = 0
            totalError = 0

            for sample in self.trainingSet.input:

                label = self.trainingSet.label[pos]
                output = self.fire(sample)
               
                error = loss.calculateError(label, output)
                grad += error * sample
                deltaWeight = self.learningRate * grad
                self.weight += deltaWeight

                totalError += error
                pos += 1

            logging.info("Iteration: %i; Error: %0.3f", iteration, -totalError)
            iteration += 1

            if round(totalError) == 0 or iteration >= self.epochs:
                learned = True



    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        output = self.fire(testInstance)

        if output > 0.5:
            return True
        else:
            return False


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def fire(self, input):
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
