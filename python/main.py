#!/usr/bin/python3

import numpy as np
from models.log_reg import LogRegModel

def get_accuracy(X_test, Y_test, function):
    """
    Returns cost history AND classification accuracy
    """
    return [0,0], 100

def model_logreg(X_train, Y_train, X_test, Y_test):
    """
    Returns:
    - cost_history_train
    - cost_history_test
    - accuracy on the test set
    """
    learning_rate = 0.01
    num_features = X.shape[0]
    num_iterations = 1500
    model = LogRegModel(num_features, learning_rate, num_iterations)
    cost_hist_train = model.train(X_train,Y_train)
    cost_hist_test, acc_test = get_accuracy(X_test, Y_test, model.classify)
    return cost_hist_train, cost_hist_test, acc_test

def model_NN(X, y, layer_dims):
    pass

def split_data(X,y):
    """
    returns X_train,Y_train, X_test, Y_test
    """
    pass
    
if __name__ == "__main__":
    # read in command line args
    # load params
    
    # call either, runLogReg, runSmallNN, runMediumNN, runLargeNN
    
    x_file = open('data/practice/X_data.csv', "r")
    y_file = open('data/practice/Y_data.csv', "r")
    
    x_data = np.genfromtxt(x_file, delimiter=',')
    y_data = np.genfromtxt(y_file, delimiter=',')

    # Format data
    x_data = x_data.T    
    y_data = y_data.reshape((1,y_data.shape[0]))

    print(x_data.shape)
    print(y_data.shape)

    
    

# Goals, run a logistic reg classifier on the cat training set
# --> develop something to compare to
# Code up an L-layer NN for the cat training set
# -> Plot learning curve
# -> start w/ small network -> go to medium -> large network
# I'll add regularization later
# If it's working, move on to SVM's!
