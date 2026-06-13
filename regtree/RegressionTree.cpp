#include "RegressionTree.hpp"
#include <numeric> 
#include <cstdlib>
#include <omp.h>

void RegressionTree::fit(const double* X, const double* y, int n_samples, int n_features) {
    if (!X || !y || n_samples <= 0) return;
    
    this->n_features_ = n_features;

    this->indices.assign(n_samples, 0);
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

void RegressionTree::predict(const double* X, double* out, int n_samples) const {
    if (!X || !out) return;
    
    for (int i = 0; i < n_samples; ++i) {
        const double* sample = &X[i * this->n_features_];
        out[i] = predict_one(sample);
    }
    
}

std::unique_ptr<Node> RegressionTree::build_tree(const double* X, 
                                                 const double* y,
                                                 int depth,
                                                 int n_samples_split, 
                                                 int mask) {
    /**
     * Uses bitmask to define splits. 
     * i-th element of array represents complex information about splits of i-th sample.
     * j-th least significant bit represents information about split at depth j. 
     * 1 represents membership of right split.
     */
    double sum = 0;
    int right_mask = 1 << depth; 
    int n_right_samples = 0;
    
    for (size_t i=0; i < this->indices.size(); i++) {
        sum += y[i] * (this->indices[i] == mask);
    }
    double mean = sum / n_samples_split;

    if (depth >= this->max_depth || n_samples_split < this->min_samples_split) {

        return std::make_unique<Node>(mean);
    }

    Split split = find_best_split(X, y, mask);

    if (split.feature_index == -1) {
        return std::make_unique<Node>(mean);
    }

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

RegressionTree::Split RegressionTree::find_best_split(const double* X, 
                                                      const double* y,
                                                      int mask) {
    double best_sse = 2e30; 
    Split best_split;
    #pragma omp parallel
    {
        double local_best_sse = 2e30;
        Split local_best_split;
        #pragma omp for nowait
        for (int f = 0; f < this->n_features_; f++) {
            for (size_t i = 0; i < this->indices.size(); i++) {
                if (this->indices[i] != mask) continue;

                double threshold = X[i * this->n_features_ + f];
                
                double sum_l = 0, sum_r = 0;
                double sum_sq_l = 0, sum_sq_r = 0;
                int count_l = 0, count_r = 0;

                for (size_t j = 0; j < this->indices.size(); j++) {
                    if (this->indices[j] != mask) continue;

                    double val = X[j * this->n_features_ + f];
                    double target = y[j];

                    if (val < threshold) {
                        sum_l += target;
                        sum_sq_l += target * target;
                        count_l++;
                    } else {
                        sum_r += target;
                        sum_sq_r += target * target;
                        count_r++;
                    }
                }

                if (count_l < this->min_samples_split || count_r < this->min_samples_split) {
                    continue;
                }

                double sse_l = sum_sq_l - (sum_l * sum_l / count_l);
                double sse_r = sum_sq_r - (sum_r * sum_r / count_r);
                double total_sse = sse_l + sse_r;

                if (total_sse < local_best_sse) {
                    local_best_sse = total_sse;
                    local_best_split.feature_index = f;
                    local_best_split.threshold = threshold;
                }
            }
        }
        #pragma omp critical
        {
            if (local_best_sse < best_sse) {
                best_sse = local_best_sse;
                best_split = local_best_split;
            }
        }
    }
    return best_split;
}

