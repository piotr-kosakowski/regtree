#ifndef REGRESSION_TREE_HPP
#define REGRESSION_TREE_HPP

#include "Node.hpp"
#include <vector>
#include <memory>

class RegressionTree {
public:
    RegressionTree(int max_depth = 10, int min_samples_split = 2) 
        : max_depth(max_depth), min_samples_split(min_samples_split) {}
    
    int max_depth;
    int min_samples_split;

    void fit(const std::vector<double>& X, const std::vector<double>& y, int n_samples, int n_features);

    std::vector<double> predict(const std::vector<double>& X, int n_samples) const;

private:
    std::unique_ptr<Node> root_;
    std::vector<int> indices;
    int n_features_ = 0;

    double predict_one(const double* x) const;

    std::unique_ptr<Node> build_tree(const std::vector<double>& X, 
                                     const std::vector<double>& y, 
                                     int depth,
                                     int n_samples_split,
                                     int mask);

    struct Split {
        int feature_index = -1;
        double threshold = 0.0;
    };

    Split find_best_split(const std::vector<double>& X, 
                          const std::vector<double>& y,
                          int mask);
};

#endif
