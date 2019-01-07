#!/usr/bin/python3

import numpy as np
from activations import sigmoid

"""
Class based approach to avoid conflicts in the main file
Uses a logistic cost function w/ sigmoid activation function
"""

class LogRegModel:

    def __init__(num_features, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.dimensions = (1, num_features)
        self.num_iters = num_iterations
        
        self.initialize_parameters()

    def train(X_train, Y_train):
        """
        Updates the parameters inside the model
        Returns the cost on the training set
        """
        m = X_train.shape[1]
        costs = []

        for i in range(self.num_iters):
            A = self.classify(X_train)
            costs.append(compute_cost(A, Y_train))
            dZ = A - Y_train # simplify using derivative of sigmoid
            dW = (1/m) * np.dot(dZ, A.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            self._update_parameters(dW, db)

        return costs

    def compute_cost(A, Y):
        return -(1/m) * ( np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))
            
    def classify(X_test):
        W = self.parameters["W"]
        b = self.parameters["b"]
        return sigmoid( np.dot(W,X_test) + b )

    def _initialize_parameters():
        self.parameters = {}
        W = np.random.randn(self.dimensions) * 0.01 # make weights start smaller increases speed of logreg
        b = 0
        self.parameters["W"] = W
        self.parameters["b"] = b

    def _update_parameters(dW, db):
        W = self.parameters["W"]
        b = self.parameters["b"]

        assert(W.shape == dW.shape)
        assert(b.shape == db.shape)
        
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db

        self.parameters["W"] = W
        self.parameters["b"] = b
