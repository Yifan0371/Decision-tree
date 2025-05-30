#include "pruner/CostComplexityPruner.hpp"
#include <cmath>

// 辅助函数：计算子树中的叶子节点数
static int countLeaves(const Node* node) {
    if (!node || node->isLeaf) return 1;
    return countLeaves(node->left.get()) + countLeaves(node->right.get());
}

double CostComplexityPruner::pruneRec(Node* n) const {
    if (n->isLeaf) {
        // 对于叶子节点，返回该节点的总误差（误差 * 样本数）
        return n->nodeMetric * n->samples;
    }
    
    // 递归剪枝子树
    double errLeft  = pruneRec(n->left.get());
    double errRight = pruneRec(n->right.get());
    
    // 子树的总误差
    double subtreeError = errLeft + errRight;
    
    // 计算子树叶子数
    int subtreeLeaves = countLeaves(n->left.get()) + countLeaves(n->right.get());
    
    // CART复杂度比较
    // 单叶成本：节点总误差 + α * 1
    double leafCost = n->nodeMetric * n->samples + alpha_;
    // 子树成本：子树总误差 + α * 子树叶子数  
    double subtreeCost = subtreeError + alpha_ * subtreeLeaves;
    
    // 如果单叶成本更低或相等，则剪枝
    if (leafCost <= subtreeCost) {
        n->isLeaf = true;
        n->left.reset();
        n->right.reset();
        return n->nodeMetric * n->samples;  // 返回单叶总误差
    }
    
    // 保留子树
    return subtreeError;
}

void CostComplexityPruner::prune(std::unique_ptr<Node>& root) const {
    if (root) pruneRec(root.get());
}