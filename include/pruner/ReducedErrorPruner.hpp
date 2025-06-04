#pragma once

#include "tree/IPruner.hpp"
#include <vector>


class ReducedErrorPruner : public IPruner {
public:
    ReducedErrorPruner(const std::vector<double>& X_val, int rowLen,
                       const std::vector<double>& y_val)
        : Xv_(X_val), D_(rowLen), yv_(y_val) {}
    void prune(std::unique_ptr<Node>& root) const override;
private:
    const std::vector<double>& Xv_;
    int D_;
    const std::vector<double>& yv_;
    double validate(Node* node) const;              
    void pruneRec(std::unique_ptr<Node>& node) const;
};
