"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        dists = euclidean_dist_squared(self.X, X_hat)
        t = X_hat.shape[0]
        assert(t == dists.shape[1])
        pred = np.zeros(t).reshape((1, t))
        
        for x_hat in range(t):
            closest_neighbors = np.argsort(dists[:, x_hat])[:self.k]
            headcount = np.bincount(self.y[closest_neighbors])
            pred[0, x_hat] = np.argmax(headcount)

        return pred
            
            

            
