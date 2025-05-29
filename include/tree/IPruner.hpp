#ifndef TREE_IPRUNER_HPP
#define TREE_IPRUNER_HPP

#include <memory>
#include "Node.hpp"

class IPruner {
public:
    virtual ~IPruner() = default;
    /** 剪枝实现（可以是预剪、后剪、或者不剪） */
    virtual void prune(std::unique_ptr<Node>& root) const = 0;
};

#endif // TREE_IPRUNER_HPP
