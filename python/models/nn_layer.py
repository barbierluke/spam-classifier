#!/usr/bin/python3

from __future__ import division
import numpy as np
from activations import sigmoid, relu, tanh
from pdb import set_trace

def relu_grad(Z):
    return Z > 0

"""
Should add the option for using relu or tanh in the middle units
"""

class NN_Model:

    predict_thresh = 0.5
    
    def __init__(self, layer_dims, alpha, num_iters):
        self.init_params(layer_dims)
        self.alpha = alpha
        self.num_iters = num_iters

    def train(self, X_train, Y_train):
        L = len(self.parameters) // 2
        costs = []

        for i in range(self.num_iters):
#            print("Iteration {}".format(i))
            AL, cache = self.feed_forward(X_train)
            cost = self.compute_cost(AL, Y_train)
            grads = self.back_prop(AL, Y_train, cache)
#            print grads
#            set_trace()
#            self.dict_to_vector(grads)
#            set_trace()
            self.update_params(grads)
            costs.append(cost)

            if i % 100 == 0:
                print("Iter {}: Cost: {}".format(i,cost))
            
        return costs

    def compute_cost(self, AL, Y):
        L = len(self.parameters) // 2
        m = Y.shape[1]
#        WL = self.parameters["W"+str(L)] For regularization
        return (-1/m) * ( np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))

    def vector_to_dict(self, vector):
        """
        Helper function with gradient checking
        Used to unflatten the giant vector into the paramters dictionary
        """
        pass

    def dict_to_vector(self, grads):
        """
        Helper function with gradient checking
        Used to flatten the parameters dictionary to a giant vector
        Meant to be used with grad_approx & grads
        """
        print grads
        grad_vector = np.array([[]])
        L = len(grads) // 2
        for i in range(1,L+1):
            W = grads['dW'+str(i)]
            b = grads['db'+str(i)]
            np.append(grad_vector, W)
            np.append(grad_vector, b)
        print grad_vector
        print grad_vector.shape
        return grad_vector

    def grad_check(self, grads, epsilon=(10**-7)):
        """
        Debug gradient descent by checking the gradients computed are roughly what they should be
        """
        # store the original params
        original_params = self.parameters
        # convert grads to a vector grads_v
        grads_v = self.dict_to_vector(grads)
        # create a new vector grad_approx using size of grads_v
        # vector = self.vector_to_dict
        # for each i in vector:
        # thetaplus = i + epsilon
        # thetaminus = i - epsilon
        # set self.params using thetaplus & get cost
        # set self.params using thetaminus & get cost
        # get the grad approx & store it in grad_approx
        # endfor
        #
        
        # get the numerator
        # get the denominator
        # divide & check for a small value

    def back_prop(self, AL, Y, cache):
        """
        Returns grads
        """
#        print(cache)
        grads = {}
        L = len(self.parameters) // 2
        m = Y.shape[1]
        dZL = AL - Y # assume sigmoid activation on output layer
        A_prev = cache['A' + str(L-1)]
        dWL = (1/m) * np.dot(dZL, A_prev.T)
        dbL = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot( cache['W'+str(L)].T, dZL)

        # add to grads
        grads['dW'+str(L)] = dWL
        grads['db'+str(L)] = dbL

        for l in reversed(range(1,L)):
            Z = cache['Z'+str(l)]
#            print(Z)
            dZ = np.multiply( dA_prev, relu_grad(Z))
#            print(dZ)
#            set_trace()
            dW = (1/m) * ( np.dot(dZ, cache['A'+str(l-1)].T) )
            db = (1/m) * ( np.sum(dZ, axis=1, keepdims=True) )
            dA_prev = np.dot( cache['W'+str(l)].T, dZ )

            # add to grads
            grads['dW'+str(l)] = dW
            grads['db'+str(l)] = db

        return grads

    def feed_forward(self, X):
        cache = {}
        cache['A0'] = X
        L = len(self.parameters) // 2
        A_prev = X
        for l in range(1, L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W,A_prev) + b
            A_prev = relu(Z)
            # add Z,A_prev, W,b to cache
            cache['W'+str(l)] = W
            cache['b'+str(l)] = b
            cache['Z'+str(l)] = Z
            cache['A'+str(l)] = A_prev

        WL = self.parameters['W' + str(L)]
        bL = self.parameters['b' + str(L)]
        ZL = np.dot(WL, A_prev) + bL
        AL = sigmoid(ZL)

        # add to cache
        cache['W'+str(L)] = WL
        cache['b'+str(L)] = bL
        cache['Z'+str(L)] = ZL
        cache['A'+str(L)] = AL
            
        return AL, cache

    def predict(self, X):
        probs, cache = self.feed_forward(X)
        return probs > self.predict_thresh

    def update_params(self, grads): # run gradient descentx
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            dW = grads['dW'+str(l)]
            db = grads['db'+str(l)]
            W = self.parameters['W'+str(l)]
            b = self.parameters['b'+str(l)]

            W = W - self.alpha * dW
            b = b - self.alpha * db

            self.parameters['W'+str(l)] = W
            self.parameters['b'+str(l)] = b

    def init_params(self, layer_dims):
        self.parameters = {}
        L = len(layer_dims)
        for l in range(1,L): # 1 - L-1
            n_prev = layer_dims[l-1]
            n_curr = layer_dims[l]
            self.parameters['W' + str(l)] = np.random.randn(n_curr,n_prev) * 0.01
            self.parameters['b' + str(l)] = np.zeros((n_curr, 1))

if __name__ == "__main__":
    layer_dims = [8, 4, 1]
    nn = NN_Model(layer_dims, 0.01, 1500)
    print(nn.parameters)                  
