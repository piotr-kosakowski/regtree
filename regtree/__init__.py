from ._core import PyRegressionTree 
import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        assert isinstance(max_depth, int) and max_depth >= 0 and max_depth < 32, "max_depth must be a integer between 0 and 31"
        assert isinstance(min_samples_split, int) and min_samples_split > 0, "min_samples_split must be an integer greater than 0"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # self._tree = None
        self._tree = PyRegressionTree(self.max_depth, self.min_samples_split)
        self._is_fitted = False
        self.n_features_ = None

    def fit(self, X, y):
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must be equal"
        assert X.flags['C_CONTIGUOUS'], "X must be a C-contiguous array"
        assert y.flags['C_CONTIGUOUS'], "y must be a C-contiguous array"
        assert np.isreal(X).all(), "X must contain only real numbers"
        assert np.isreal(y).all(), "y must contain only real numbers"
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        self._tree.fit(X[::1].ravel(), y[::1], X.shape[0], X.shape[1])
        self.n_features_ = X.shape[1]
        self._is_fitted = True

    def predict(self, X):
        # if not self._is_fitted:
        #     raise ValueError("The model has not been fitted yet.")
        assert self._is_fitted, "The model has not been fitted yet."
        assert X.ndim == 2, "X must be a 2D array"
        assert X.shape[1] == self.n_features_, f"X must have {self.n_features_} features"
        assert X.flags['C_CONTIGUOUS'], "X must be a C-contiguous array"
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._tree.predict(X[::1].ravel(), X.shape[0])