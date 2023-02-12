"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

from utils import euclidean_dist_squared

class NKNN:
    ensemble_size = None


# augmented KNN model that can also fit the best (lowest error) k to its dataset
class KNN:
    X = None
    y = None
    best_k = None

    # valid ks should be a range, which includes all the k's the model should consider when finding its best k
    def __init__(self, valid_ks):
        self.valid_ks = valid_ks

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    # uses cross validation to set self.best_k to whatever 
    def find_best_k(self):
        # NOTE: this program will automatically do the HIGHEST possible number of cv folds while allowing each validation set to have at least min_items_per_fold examples
        # I set this hyper-hyper parameter to 20, as this 'feels' like a good number for this dataset, for lack of a better reason
        min_items_per_fold = 20
        ks = list(self.valid_ks)

        n = int(self.X.shape[0])
        num_folds = n / min_items_per_fold
        fold_size = int(n / num_folds)
        leftovers = int(n % num_folds)

        cv_accs = np.zeros(len(ks))
        for k in ks:
            # clear errors for this iteration
            errors = np.empty(num_folds, dtype=float)
            for i in range(num_folds):
                # clear mask
                mask = np.zeros(n, dtype=bool)

                # set mask for this iteration
                i_0 = (i * fold_size)
                i_1 = ((i + 1) * fold_size)
                # in case n is not evenly divisible by the number of fold we are doing,
                #     append the leftover data to the end of the last validation set
                if i == num_folds - 1:
                    i_1 = i_1 + leftovers
                mask[i_0:i_1] = True

                # get X_validate and X_train from X using mask
                X_validate = X[mask, :]
                y_validate = y[mask]
                X_train = X[~mask, :]
                y_train = y[~mask]

                # train and validate on this fold, and store validation error in errors
                model = KNN(k=k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_validate)
                err = np.abs(y_pred - y_validate)
                
                err = np.mean(err)
                errors[i] = err
            
            # store average error for k across the folds
            cv_accs[ks.index(k)] = np.mean(errors)
        self.best_k = ks[np.argmin(cv_accs)]

    def predict(self, X_hat):
        if self.best_k == None:
            self.find_best_k()
        
        dists = euclidean_dist_squared(self.X, X_hat)
        t = X_hat.shape[0]
        assert(t == dists.shape[1])
        pred = np.zeros(t).reshape((1, t))
        
        for x_hat in range(t):
            closest_neighbors = np.argsort(dists[:, x_hat])[:self.best_k]
            headcount = np.bincount(self.y[closest_neighbors])
            pred[0, x_hat] = np.argmax(headcount)

        return pred
            
            

            
