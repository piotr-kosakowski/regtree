from ._core import PyRegressionTree 

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = PyRegressionTree(self.max_depth, self.min_samples_split)

    def fit(self, X, y):
        self.tree.fit(X[::1], y[::1], X.shape[0], X.shape[1])

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The model has not been fitted yet.")
        return self.tree.predict(X[::1], X.shape[0])