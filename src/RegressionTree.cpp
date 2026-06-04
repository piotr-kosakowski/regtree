#include "RegressionTree.hpp"
#include <numeric> 

void RegressionTree::fit(const std::vector<double>& X, const std::vector<double>& y, int n_samples, int n_features) {
    if (X.empty() || y.empty() || n_samples <= 0) return;
    
    this->n_features_ = n_features;

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0); 

    this->root_ = build_tree(X, y, indices, 0);
}

double RegressionTree::predict_one(const double* x) const {
    if (!this->root_) return 0.0;

    Node* current = this->root_.get();
    while (!current->is_leaf) {
        if (x[current->feature_index] < current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    return current->predicted_value;
}

std::vector<double> RegressionTree::predict(const std::vector<double>& X, int n_samples) const {
    std::vector<double> predictions;
    predictions.reserve(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        const double* sample = &X[i * this->n_features_];
        predictions.push_back(predict_one(sample));
    }
    
    return predictions;
}

std::unique_ptr<Node> RegressionTree::build_tree(const std::vector<double>& X, 
                                                 const std::vector<double>& y, 
                                                 std::vector<int>& indices,
                                                 int depth) {
    double sum = 0;
    for (int idx : indices) {
        sum += y[idx];
    }
    double mean = sum / indices.size();

    if (depth >= this->max_depth || indices.size() < (size_t)this->min_samples_split) {
        return std::make_unique<Node>(mean);
    }

    Split split = find_best_split(X, y, indices);


    auto node = std::make_unique<Node>();
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;

    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        if (X[idx * this->n_features_ + split.feature_index] < split.threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    node->left = build_tree(X, y, left_indices, depth + 1);
    node->right = build_tree(X, y, right_indices, depth + 1);

    return node;
}

RegressionTree::Split RegressionTree::find_best_split(const std::vector<double>& X, 
                                                      const std::vector<double>& y,
                                                      const std::vector<int>& indices) {
    // TODO: Logic for finding the split that reduces MSE the most
    return Split();
}
