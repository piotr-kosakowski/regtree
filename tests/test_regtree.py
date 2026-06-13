import pytest
import numpy as np
from regtree import DecisionTreeRegressor

def test_constant_value():
    X = np.random.rand(100, 5)
    y = np.ones(100)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, 1), "Predictions should be all ones for constant target"

def test_average_value():
    X = np.random.rand(100, 5)
    y = np.arange(100)
    model = DecisionTreeRegressor(max_depth=0, min_samples_split=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, np.mean(y))

def test_not_contiguous_array():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.fit(X[::2], y[::2])

def test_non_real_values():
    X = np.random.rand(100, 5) + 1j
    y = np.random.rand(100) + 1j
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.fit(X, y)

def test_mismatched_samples():
    X = np.random.rand(100, 5)
    y = np.random.rand(99)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.fit(X, y)

def test_invalid_X_dimension():
    X = np.random.rand(100)
    y = np.random.rand(100)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.fit(X, y)

def test_invalid_y_dimension():
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.fit(X, y)

def test_invalid_predict_X_dimension():
    X = np.random.rand(100, 5)
    X_test = np.random.rand(5, 7)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    model.fit(X, np.random.rand(100))
    with pytest.raises(AssertionError):
        model.predict(X_test)

def test_predict_before_fit():
    X = np.random.rand(100, 5)
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    with pytest.raises(AssertionError):
        model.predict(X)

def test_invalid_max_depth():
    with pytest.raises(AssertionError):
        DecisionTreeRegressor(max_depth=-1, min_samples_split=2)

def test_invalid_min_samples_split():
    with pytest.raises(AssertionError):
        DecisionTreeRegressor(max_depth=3, min_samples_split=0)

