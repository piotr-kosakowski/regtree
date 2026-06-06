#include "RegressionTree.hpp"
#include <numeric> 

void RegressionTree::fit(const std::vector<double>& X, const std::vector<double>& y, int n_samples, int n_features) {
    if (X.empty() || y.empty() || n_samples <= 0) return;
    
    this->n_features_ = n_features;

    // std::vector<int> indices(n_samples);
    // std::iota(indices.begin(), indices.end(), 0); 
    std::fill(this->indices.begin(), this->indices.end(), 0);

    this->root_ = build_tree(X, y, 0, n_samples, 0);
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
                                                 int depth,
                                                 int n_samples_split, 
                                                 int mask) {
    double sum = 0;
    int right_mask = 1 << (depth-1);
    int n_right_samples = 0;
    for (size_t i=0; i < this->indices.size(); i++) {
        sum += y[i] * (this->indices[i] == mask);
    }
    double mean = sum / n_samples_split;

    if (depth >= this->max_depth || n_samples_split < this->min_samples_split) {

        return std::make_unique<Node>(mean);
    }

    Split split = find_best_split(X, y, mask);


    auto node = std::make_unique<Node>();
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;

    for (size_t i=0; i < this->indices.size(); i++) {
        if (this->indices[i] != mask) continue;
        if (X[i * this->n_features_ + split.feature_index] >= split.threshold) {
            this->indices[i] += right_mask;
            n_right_samples++;
        }
    }


    node->left = build_tree(X, y, depth + 1, n_samples_split - n_right_samples, mask); 
    node->right = build_tree(X, y, depth + 1, n_right_samples, mask+right_mask); 

    return node;
}

RegressionTree::Split RegressionTree::find_best_split(const std::vector<double>& X, 
                                                      const std::vector<double>& y,
                                                      int mask ) {
    // TODO: Logic for finding the split that reduces MSE the most
    return Split();
}
