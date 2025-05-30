#ifndef COST_COMPLEXITY_PRUNER_HPP
#define COST_COMPLEXITY_PRUNER_HPP
#include "tree/IPruner.hpp"

/** CART 复杂度后剪枝：剪掉 α 带来的最小 Cost Complexity 的子树 */
class CostComplexityPruner : public IPruner {
public:
    explicit CostComplexityPruner(double alpha) : alpha_(alpha) {}
    void prune(std::unique_ptr<Node>& root) const override;
private:
    double alpha_;
    double pruneRec(Node* node) const;   // 返回子树误差(加α·T)
};
#endif