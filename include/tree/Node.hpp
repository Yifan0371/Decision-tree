#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

#include <memory>
#include <cstddef>

struct Node {
    int featureIndex = -1;
    double threshold   = 0.0;
    double prediction  = 0.0;
    bool   isLeaf      = false;
    double metric      = 0.0;
    size_t samples     = 0;
    std::unique_ptr<Node> left  = nullptr;
    std::unique_ptr<Node> right = nullptr;
};

#endif // TREE_NODE_HPP
