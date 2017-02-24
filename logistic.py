""" Methods for doing logistic regression."""

import numpy as np
from utils import *


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    data_colum = np.ones((data.shape[0],1))
    data_new = np.append(data, data_colum, axis=1)
    z = np.dot(data_new, weights)
    y = 1/(1+np.exp(-z))

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x: 1 vector of probabilities.
    Outputs
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function

    ce = -np.dot(targets.transpose(), np.log(y)) - np.dot((1-targets.transpose()),np.log(1-y))
    ce = ce[0][0]

    #counting how many of the predictions are correct
    counter = 0
    for m,n in zip(y, targets):
        if m>=0.5 and n == 1:
            counter += 1
        elif m<0.5 and n == 0:
            counter += 1

    #correct fraction = total correctly predicted/total target data set
    frac_correct = counter*1.0/len(targets)*1.0

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        """
        data_row = np.ones((1, data.transpose().shape[1]))
        data_new = np.append(data.transpose(), data_row, axis=0)
        z = np.dot(weights.transpose(), data_new)
        """
        #make data_new N x M+1
        #z is N x 1
        data_colum = np.ones((data.shape[0], 1))
        data_new = np.append(data, data_colum, axis=1)

        f, _ = evaluate(targets, y)
        df = np.dot(data_new.transpose(), y-targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)
    M = data.shape[1]
    data_colum = np.ones((data.shape[0], 1))
    data_new = np.append(data, data_colum, axis=1)
    #z = np.dot(data_new, weights)

    weight_decay = hyperparameters["weight_decay"]
    constant = np.asscalar(weight_decay/2.0*(np.dot(weights[:-1].transpose(),weights[:-1])) - M/2.0*np.log(weight_decay/(2*np.pi)))
    f, _ = evaluate(targets, y)
    f = f + constant
    regularization = weight_decay*(weights)
    regularization[M] = 0
    df = np.dot(data_new.transpose(), y - targets) + regularization

    return f, df
