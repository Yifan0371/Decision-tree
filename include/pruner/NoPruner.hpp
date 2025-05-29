#ifndef NO_PRUNER_HPP
#define NO_PRUNER_HPP

#include "../tree/IPruner.hpp"

class NoPruner : public IPruner {
public:
    void prune(std::unique_ptr<Node>& root) const override {
        // 不做任何剪枝
    }
};

#endif // NO_PRUNER_HPP
