from model.classifier import Classifier
import numpy as np
from util.activation_functions import Activation

class Perceptron(Classifier):
    '''
    A digit-7 recognizer based on perceptron algorithm 
    '''
    def __init__(self, train, valid, test, learningRate = 0.01, epochs=50):
        
        self.learningRate = learningRate
        self.epochs = epochs
        
        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        # Initialize the weight vector with small values
        self.weight= np.random.rand(self.trainingSet.input.shape[1],1)/1000  # random values around 0 and 0.1
    
        
    def train(self, trainingSet, validationSet):
        # Here you have to implement the Perceptron Learning Algorithm
        return self.fire(testInstance)
    
    def classify(self, testInstance):
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7, False otherwise
        pass
    
    def evaluate(self, test):
        # One you can classify an instance, just use map for all of the test set.
        return map(self.classify, test.input)
    
    def fire(self, input):
        return Activation.sign(np.dot(np.array(input),self.weight))
