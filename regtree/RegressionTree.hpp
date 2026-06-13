#ifndef REGRESSION_TREE_HPP
#define REGRESSION_TREE_HPP

#include "Node.hpp"
#include <vector>
#include <memory>

/**
 * @brief Represents whole regression Tree for numercial features.
 * 
 * This class implements a greedy recursive algortihm that
 * minimizes SSE at each split.
 */
class RegressionTree {
public:
    /**
     * @brief Constructor for a new Regression Tree object.
     * @param max_depth Maximum depth of tree (0 = root only). Must be < 32.
     * @param min_samples_split  Minimum samples required to attempt a split.
     */
    RegressionTree(int max_depth = 10, int min_samples_split = 2) 
        : max_depth(max_depth), min_samples_split(min_samples_split) {}
    
    int max_depth;
    int min_samples_split;
    /**
     * @brief  Fits the tree to a training dataset.
     * @param X Flat, C-contigious 1D array of features (size: n_samples * n_features). 
     * @param y Flat, C-contigious 1D array of target va;ues (size: n_samples).
     * @param n_samples Number of samples (rows).
     * @param n_features Number of features (columns).
     */
    void fit(const double* X, const double* y, int n_samples, int n_features);

    /**
    * @brief Predicts target values for multiple samples.
    * @param X Flat 1D array of features to predict.
    * @param out Pointer to the array where predictions will be stored.
    * @param n_samples Number of samples to predict.
    */
    void predict(const double* X, double* out, int n_samples) const;

private:
    std::unique_ptr<Node> root_;
    std::vector<int> indices; // Bitmask vector used for in-place partitioning.
    int n_features_ = 0;

    /** @brief Traverses the tree for a single sample and returns final leaf's value. */
    double predict_one(const double* x) const;

    /**
     * @brief Internal recursive function to build the tree. 
     * 
     * @param mask The current path identifier (bit pattern) for this node.
     * @details The 'indices' vector stores a bitmask for each sample.
     * The j-th bit represents the split decision at depth j: 
     * 0 = left child, 1 = right child. 
     * A sample belongs to this node only if its bitmask matches the 'mask' parameter.
     * */
    std::unique_ptr<Node> build_tree(const double* X, 
                                     const double* y, 
                                     int depth,
                                     int n_samples_split,
                                     int mask);
    /** @brief Helper structure to store the best split found. */
    struct Split {
        int feature_index = -1;
        double threshold = 0.0;
    };

    /** @brief Checks possible splits to find the one, that minimizes SSE */
    Split find_best_split(const double* X, 
                          const double* y,
                          int mask);
};

#endif
