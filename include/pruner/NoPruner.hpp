#pragma once

#include "../tree/IPruner.hpp"

class NoPruner : public IPruner {
public:
    void prune(std::unique_ptr<Node>& ) const override {
        
    }
};
