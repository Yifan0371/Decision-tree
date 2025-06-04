#pragma once

#include <memory>
#include "Node.hpp"

class IPruner {
public:
    virtual ~IPruner() = default;
    
    virtual void prune(std::unique_ptr<Node>& root) const = 0;
};
