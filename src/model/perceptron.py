from model.classifier import Classifier
import numpy as np
from util.activation_functions import Activation


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1], 1)/1000

    def train(self, trainingSet, validationSet):
        """Train the perceptron with the perceptron learning algorithm."""
        # TODO: Here you have to implement the Perceptron Learning Algorithm
        return self.fire(testInstance)

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
        # TODO: Here you have to implement the classification for one instance
        pass

    def evaluate(self, test):
        # One you can classify an instance, just use map for all of the test
        # set.
        return map(self.classify, test.input)

    def fire(self, input):
        return Activation.sign(np.dot(np.array(input), self.weight))
