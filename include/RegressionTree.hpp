#ifndef REGRESSION_TREE_HPP
#define REGRESSION_TREE_HPP

#include "Node.hpp"
#include <vector>
#include <memory>


class RegressionTree {
public:
    struct Params {
        int max_depth = 10;
        int min_samples_split = 2;
    };

    RegressionTree(Params params = Params()) : params_(params) {}


    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);


    double predict_one(const std::vector<double>& x) const;

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    Params params_;
    std::unique_ptr<Node> root_;

    std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>& X, 
                                     const std::vector<double>& y, 
                                     int depth);

    struct Split {
        int feature_index = -1;
        double threshold = 0.0;
        double mse_reduction = -1.0;
    };
    Split find_best_split(const std::vector<std::vector<double>>& X, 
                          const std::vector<double>& y);
};

#endif
