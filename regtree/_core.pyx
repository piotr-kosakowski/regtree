import numpy as np
cimport numpy as cnp

cdef extern from "RegressionTree.hpp":
    cdef cppclass RegressionTree:
        RegressionTree(int max_depth, int min_samples_split) except +

        void fit(const double* X, const double* y, int n_samples, int n_features) except +

        void predict(const double* X, double* out, int n_samples) except +

cdef class PyRegressionTree:
    cdef RegressionTree* _cpp_tree

    def __cinit__(self, int max_depth=10, int min_samples_split=2):
        self._cpp_tree = new RegressionTree(max_depth, min_samples_split)

    def __dealloc__(self):
        if self._cpp_tree != NULL:
            del self._cpp_tree

    def fit(self, double[:] X, double[:] y, int n_samples, int n_features):
        self._cpp_tree.fit(&X[0], &y[0], n_samples, n_features)

    def predict(self, double[:] X, int n_samples):
        out = np.empty(n_samples, dtype=np.float64)
        cdef double[:] out_view = out
        self._cpp_tree.predict(&X[0], &out_view[0], n_samples)
        return out
