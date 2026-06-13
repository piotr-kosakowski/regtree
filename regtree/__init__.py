from ._core import PyRegressionTree 
import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        """
        
        """
        assert isinstance(max_depth, int) and max_depth > 0 and max_depth < 32, "max_depth must be a positive integer between 1 and 31"
        assert isinstance(min_samples_split, int) and min_samples_split > 1, "min_samples_split must be an integer greater than 1"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = PyRegressionTree(self.max_depth, self.min_samples_split)

    def fit(self, X, y):
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must be equal"
        assert X.flags['C_CONTIGUOUS'], "X must be a C-contiguous array"
        assert y.flags['C_CONTIGUOUS'], "y must be a C-contiguous array"
        assert np.isreal(X).all(), "X must contain only real numbers"
        assert np.isreal(y).all(), "y must contain only real numbers"
        self.tree.fit(X[::1].ravel(), y[::1], X.shape[0], X.shape[1])

    def predict(self, X):
        assert X.ndim == 2, "X must be a 2D array"
        assert X.flags['C_CONTIGUOUS'], "X must be a C-contiguous array"
        if self.tree is None:
            raise ValueError("The model has not been fitted yet.")
        return self.tree.predict(X[::1].ravel(), X.shape[0])