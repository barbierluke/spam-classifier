#!/usr/bin/python3

from __future__ import division
import random
import numpy as np
import argparse
from models.log_reg import LogRegModel
from models.nn_layer import NN_Model
from pdb import set_trace

def get_accuracy(X_test, Y_test, predict_func):
    """
    Returns classification accuracy
    """
    Y_predictions = predict_func(X_test)
    correct = np.sum(Y_predictions == Y_test)
    return (correct / Y_test.shape[1]) * 100

def split_data(X,Y):
    """
    returns X_train,Y_train, X_test, Y_test
    Eventually we'll also creave a cv to tune hyperparams
    """
    test_percentage = 0.2
    train_percentage = 1 - test_percentage
    m = X.shape[1]

    test_indices = random.sample(range(m), int(0.2 * m))
    train_indices = [x for x in list(range(m)) if x not in test_indices]

    X_train = X[:, train_indices]
    Y_train = Y[:, train_indices]
    X_test = X[:, test_indices]
    Y_test = Y[:, test_indices]

    return X_train, Y_train, X_test, Y_test

def model_logreg(X_train, Y_train, X_test, Y_test):
    """
    Returns:
    - cost_history_train
    - accuracy on the test set
    """
    print("Modeling Logistic Regression")
    
    num_features = X_train.shape[0]    
    learning_rate = 0.05
    num_iterations = 1500
    regularization_constant = 0

    model = LogRegModel(num_features, learning_rate, num_iterations, regularization_constant)
    cost_hist_train = model.train(X_train,Y_train)

    accuracy_trainset = get_accuracy(X_train, Y_train, model.predict)    
    accuracy_testset = get_accuracy(X_test, Y_test, model.predict)

    return accuracy_trainset, accuracy_testset

def model_nn(X_train, Y_train, X_test, Y_test, layer_dims):
    print("Modeling Neural Network w/ layers: {}".format(layer_dims))
    alpha = 0.05
    num_iters = 1500

    model = NN_Model(layer_dims, alpha, num_iters)
    costs = model.train(X_train, Y_train)
    acc_test = get_accuracy(X_test, Y_test, model.predict)
    acc_train = get_accuracy(X_train, Y_train, model.predict)
    
    return acc_train, acc_test
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control which models to train on the spam data set")
    parser.add_argument('model', help="Train a logistic regression model")
    args = parser.parse_args()
    model = args.model
    # call either, runLogReg, runSmallNN, runMediumNN, runLargeNN
    
    x_file = open('data/practice/X_data.csv', "r")
    y_file = open('data/practice/Y_data.csv', "r")
    
    x_data = np.genfromtxt(x_file, delimiter=',')
    y_data = np.genfromtxt(y_file, delimiter=',')

    # Format data
    X = x_data.T    
    Y = y_data.reshape((1,y_data.shape[0]))
    X_train, Y_train, X_test, Y_test = split_data(X,Y)

    if model == "l":
        acc_train_log, acc_test_log = model_logreg(X_train, Y_train, X_test, Y_test)
        print("Logistic Regression Accuracy on Train Set: {}".format(acc_train_log))
        print("Logistic Regression Accuracy on Test Set: {}".format(acc_test_log))
    elif model == "nn":
        layer_dims = [X_train.shape[0],20,5,1]
        acc_train_nn, acc_test_nn = model_nn(X_train, Y_train, X_test, Y_test, layer_dims)
        print("NN Accuracy on Train Set: {}".format(acc_train_nn))
        print("NN Accuracy on Test Set: {}".format(acc_test_nn))
    else:
        print("Unrecognized cmd argument, pass in either 'nn' or 'l'")


