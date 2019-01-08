#!/usr/bin/python3

from __future__ import division
import numpy as np
from activations import sigmoid
from pdb import set_trace

"""
Class based approach to avoid conflicts in the main file
Uses a logistic cost function w/ sigmoid activation function
"""

class LogRegModel:

    prediction_threshold = 0.5
    
    def __init__(self, num_features, learning_rate, num_iterations, regularization_constant):
        self.learning_rate = learning_rate
        self.dimensions = (1, num_features)
        self.num_iters = num_iterations
        self.reg_constant = regularization_constant
        
        self._initialize_parameters()

    def train(self, X_train, Y_train):
        """
        Updates the parameters inside the model
        Returns the cost on the training set
        """
        print("Training LogReg")
        m = X_train.shape[1]
        costs = []

        for i in range(self.num_iters):
            A = self._activate(X_train)
            cost = self.compute_cost(A, Y_train)
            dZ = A - Y_train # simplify using derivative of sigmoid
            dW = (1/m) * ( np.dot(dZ, X_train.T) + self.reg_constant * self.parameters['W'])
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            self._update_parameters(dW, db)

            if i % 100 == 0:
                print("Iter: {} \tCost: {}".format(i,cost))

            costs.append(cost)
                    
        return costs

    def compute_cost(self, A, Y):
        m = Y.shape[1]
        W = self.parameters["W"]
        return (-1/m) * ( np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T - self.reg_constant * np.dot(W,W.T) ))
            
    def _activate(self,X):
        W = self.parameters["W"]
        b = self.parameters["b"]
        return sigmoid( np.dot(W,X) + b )
    
    def predict(self, X):
        probabilities = self._activate(X)
        return probabilities > self.prediction_threshold

    def reset_params(self):
        self._initialize_parameters()

    def _initialize_parameters(self):
        self.parameters = {}
        # W = np.random.randn(self.dimensions[0],self.dimensions[1]) * 0.01 # make weights start smaller increases speed of logreg
        W = np.zeros((self.dimensions[0], self.dimensions[1]))
        b = np.array([[0.0]])
        self.parameters["W"] = W
        self.parameters["b"] = b

    def _update_parameters(self, dW, db):
        W = self.parameters["W"]
        b = self.parameters["b"]

        assert(W.shape == dW.shape)
        assert(b.shape == db.shape)        
        
        Wnew = W - self.learning_rate * dW
        bnew = b - self.learning_rate * db

        self.parameters["W"] = Wnew
        self.parameters["b"] = bnew
