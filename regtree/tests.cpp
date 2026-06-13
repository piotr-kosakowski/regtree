#include "RegressionTree.hpp"
#include <cassert>
#include <iostream>
#include <vector>


void test_empty_data() {
    std::cout << "Running test_empty_data..." << std::endl;
    RegressionTree tree;
    // double* X;
    // double* y;
    
    tree.fit(nullptr, nullptr, 0, 1);
    
    tree.predict(nullptr,nullptr, 0);
    // assert(!pred);
    std::cout << "test_empty_data: PASSED" << std::endl;
}

void test_constant_target() {
    std::cout << "Running test_constant_target..." << std::endl;
    RegressionTree tree(10, 2);
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {42.0, 42.0, 42.0, 42.0};
    double results[4];
    
    
    tree.fit(X, y, 4, 1);
    
    // std::vector<double> pred = tree.predict(X, 4);
    // for(double p : pred) {
    //     assert(p == 42.0);
    // }

    tree.predict(X, results, 4);
    for(size_t i=0; i<4; i++) {
        assert(results[i] == 42.0);
    }
    std::cout << "test_constant_target: PASSED" << std::endl;
}

void test_single_split() {
    std::cout << "Running test_single_split..." << std::endl;
    RegressionTree tree(1, 2); 
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {10.0, 10.0, 20.0, 20.0};
    double results[2];

    
    tree.fit(X, y, 4, 1);
    
    double test_X[] = {1.5, 3.5};
    tree.predict(test_X, results, 2);
    
    assert(results[0] == 10.0);
    assert(results[1] == 20.0);
    std::cout << "test_single_split: PASSED" << std::endl;
}

void test_min_samples_split() {
    std::cout << "Running test_min_samples_split..." << std::endl;
    RegressionTree tree(2, 100); 
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {10.0, 20.0, 30.0, 40.0}; 
    double results[4];

    
    tree.fit(X, y, 4, 1);
    
    // std::vector<double> pred = tree.predict(X, 4);
    // for(double p : pred) {
    //     assert(p == 25.0);
    // }
    tree.predict(X, results, 4);
    for(size_t i=0; i<4; i++) {
        assert(results[i] == 25.0);
    }
    std::cout << "test_min_samples_split: PASSED" << std::endl;
}

void test_depth() {
    std::cout << "Running test_depth..." << std::endl;
    RegressionTree tree(0, 5); 
    double X[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {10.0, 20.0, 30.0, 40.0}; 
    double results[4];

    
    tree.fit(X, y, 4, 1);
    
    tree.predict(X, results, 4);
    for(size_t i=0; i<4; i++) {
        assert(results[i] == 25.0);
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
