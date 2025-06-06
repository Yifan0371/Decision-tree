#include "pruner/CostComplexityPruner.hpp"
#include <cmath>

// 辅助函数：计算子树中的叶子节点数
static int countLeaves(const Node* node) {
    if (!node || node->isLeaf) return 1;
    return countLeaves(node->getLeft()) + countLeaves(node->getRight());
}

double CostComplexityPruner::pruneRec(Node* n) const {
    if (n->isLeaf) {
        // 对于叶子节点，返回该节点的总误差（误差 * 样本数）
        // 使用getNodePrediction()获取剪枝预测值对应的误差
        return n->metric * n->samples;
    }
    
    // 递归剪枝子树
    double errLeft  = pruneRec(n->getLeft());
    double errRight = pruneRec(n->getRight());
    
    // 子树的总误差
    double subtreeError = errLeft + errRight;
    
    // 计算子树叶子数
    int subtreeLeaves = countLeaves(n->getLeft()) + countLeaves(n->getRight());
    
    // CART复杂度比较
    // 单叶成本：节点总误差 + α * 1
    double leafCost = n->metric * n->samples + alpha_;
    // 子树成本：子树总误差 + α * 子树叶子数  
    double subtreeCost = subtreeError + alpha_ * subtreeLeaves;
    
    // 如果单叶成本更低或相等，则剪枝
    if (leafCost <= subtreeCost) {
        // 获取剪枝后的预测值
        double nodePred = 0.0;
        if (n->isLeaf) {
            nodePred = n->getPrediction();
        } else {
            // 对于内部节点，需要重新计算均值作为叶节点预测值
            // 这里简化处理，使用当前节点的样本均值
            nodePred = n->getNodePrediction();
        }
        
        n->makeLeaf(nodePred, nodePred);
        return n->metric * n->samples;  // 返回单叶总误差
    }
    
    // 保留子树
    return subtreeError;
}

void CostComplexityPruner::prune(std::unique_ptr<Node>& root) const {
    if (root) pruneRec(root.get());
}