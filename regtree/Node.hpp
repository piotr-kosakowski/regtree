#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
/**
 * @brief Represent a single node in regression tree.
 * 
 * A node can be either an internal one with a split rule
 * or a leaf with a predicted value. 
 */

struct Node {
    // Split information for internal nodes
    int feature_index = -1;
    double threshold = 0.0;
    
    // Leaf information
    bool is_leaf = false;
    double predicted_value = 0.0;
    
    // Pointers to child nodes for values < threshold (left) and > threshold (right)
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    /** @brief Default constructor for internal nodes. */
    Node() = default;
    /**
     * @brief Constructor for leaf nodes.
     * @param val The constant value this leaf will predict.
     */
    explicit Node(double val) : is_leaf(true), predicted_value(val) {}
};

#endif 
