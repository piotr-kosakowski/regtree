#ifndef NODE_HPP
#define NODE_HPP

#include <memory>


struct Node {
    int feature_index = -1;
    double threshold = 0.0;
    
    bool is_leaf = false;
    double predicted_value = 0.0;
    
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node() = default;
    explicit Node(double val) : is_leaf(true), predicted_value(val) {}
};

#endif 
