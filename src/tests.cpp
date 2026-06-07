#include "RegressionTree.hpp"
#include <cassert>
#include <iostream>
#include <vector>


void test_empty_data() {
    std::cout << "Running test_empty_data..." << std::endl;
    RegressionTree tree;
    std::vector<double> X;
    std::vector<double> y;
    
    tree.fit(X, y, 0, 1);
    
    std::vector<double> pred = tree.predict(X, 0);
    assert(pred.empty());
    std::cout << "test_empty_data: PASSED" << std::endl;
}

void test_constant_target() {
    std::cout << "Running test_constant_target..." << std::endl;
    RegressionTree tree(10, 2);
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {42.0, 42.0, 42.0, 42.0};
    
    tree.fit(X, y, 4, 1);
    
    std::vector<double> pred = tree.predict(X, 4);
    for(double p : pred) {
        assert(p == 42.0);
    }
    std::cout << "test_constant_target: PASSED" << std::endl;
}

void test_single_split() {
    std::cout << "Running test_single_split..." << std::endl;
    RegressionTree tree(1, 2); 
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {10.0, 10.0, 20.0, 20.0};
    
    tree.fit(X, y, 4, 1);
    
    std::vector<double> test_X = {1.5, 3.5};
    std::vector<double> pred = tree.predict(test_X, 2);
    
    assert(pred[0] == 10.0);
    assert(pred[1] == 20.0);
    std::cout << "test_single_split: PASSED" << std::endl;
}

void test_min_samples_split() {
    std::cout << "Running test_min_samples_split..." << std::endl;
    RegressionTree tree(2, 100); 
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {10.0, 20.0, 30.0, 40.0}; 
    
    tree.fit(X, y, 4, 1);
    
    std::vector<double> pred = tree.predict(X, 4);
    for(double p : pred) {
        assert(p == 25.0);
    }
    std::cout << "test_min_samples_split: PASSED" << std::endl;
}

void test_depth() {
    std::cout << "Running test_depth..." << std::endl;
    RegressionTree tree(0, 5); 
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {10.0, 20.0, 30.0, 40.0}; 
    
    tree.fit(X, y, 4, 1);
    
    std::vector<double> pred = tree.predict(X, 4);
    for(double p : pred) {
        assert(p == 25.0);
    }
    std::cout << "test_depth: PASSED" << std::endl;
}

int main() {
    try {
        test_empty_data();
        test_constant_target();
        test_single_split();
        test_min_samples_split();
        
        std::cout << "\nALL C++ CORE TESTS PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
