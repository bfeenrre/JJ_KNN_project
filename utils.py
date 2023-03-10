import argparse
import numpy as np


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T,
    #        containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD
    registers and multiple cores (and faster matrix-multiplication algorithms),
    vector and matrix operations in Numpy will often be several times faster
    than if you implemented them yourself in a fast language like C. The
    following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in
    D, then element D[i,j] gives the squared Euclidean distance between training
    point i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """
    # for reference, sklearn.metrics.pairwise.euclidean_distances
    # does this but a little bit nicer; this code is just here so you can
    # easily see that it's not doing anything actually very complicated

    X_norms_sq = np.sum(X ** 2, axis=1)
    Xtest_norms_sq = np.sum(Xtest ** 2, axis=1)
    dots = X @ Xtest.T

    return X_norms_sq[:, np.newaxis] + Xtest_norms_sq[np.newaxis, :] - 2 * dots



################################################################################
# Helpers for setting up the command-line interface

_funcs = {}

def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(task):
    if task not in _funcs:
        raise ValueError(f"unknown command {task}")
    return _funcs[task]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(_funcs.keys()) + ["all"])
    args = parser.parse_args()
    if (args.task == "all"):
        for t in sorted(_funcs.keys()):
            start = f"== {t} "
            print("\n" + start + "=" * (80 - len(start)))
            run(t)
    else:
        return run(args.task)
