#pragma once

#include "tree/IPruner.hpp"


class CostComplexityPruner : public IPruner {
public:
    explicit CostComplexityPruner(double alpha) : alpha_(alpha) {}
    void prune(std::unique_ptr<Node>& root) const override;
private:
    double alpha_;
    double pruneRec(Node* node) const;   
};
