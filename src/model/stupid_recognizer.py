from model.classifier import Classifier
from random import random


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.
    """

    def __init__(self, train, valid, test, byChance=0.5):

        self.byChance = byChance

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

    def train(self):
        # Do nothing
        pass

    def classify(self, testInstance):
        # byChance is the probability of being correctly recognized
        return random() < self.byChance

    def evaluate(self):
        return list(map(self.classify, self.testSet.input))
        #return map(lambda: random() < self.byChance(), test.X)
