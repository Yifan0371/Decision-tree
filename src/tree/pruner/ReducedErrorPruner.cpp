#include "pruner/ReducedErrorPruner.hpp"
#include <cmath>

double ReducedErrorPruner::validate(Node* n) const {
    double mse = 0.0;
    for (size_t i = 0; i < yv_.size(); ++i) {
        const double* sample = &Xv_[i * D_];
        Node* cur = n;
        while (!cur->isLeaf) {
            cur = (sample[cur->getFeatureIndex()] <= cur->getThreshold())
                    ? cur->getLeft() : cur->getRight();
        }
        double diff = yv_[i] - cur->getPrediction();
        mse += diff * diff;
    }
    return mse / yv_.size();
}

void ReducedErrorPruner::pruneRec(std::unique_ptr<Node>& n) const {
    if (!n || n->isLeaf) return;
    
    // 先递归剪枝子树
    pruneRec(n->leftChild);
    pruneRec(n->rightChild);
    
    // 备份当前状态
    bool oldIsLeaf = n->isLeaf;
    double oldPred = n->isLeaf ? n->getPrediction() : 0.0;
    
    // 备份子节点（移动到临时变量）
    auto leftBackup  = std::move(n->leftChild);
    auto rightBackup = std::move(n->rightChild);

    // 尝试将当前节点变为叶节点
    // 计算叶节点预测值（这里使用节点的平均值）
    double leafPrediction = n->getNodePrediction();
    if (leafPrediction == 0.0) {
        // 如果没有存储节点预测值，使用当前的预测值
        leafPrediction = oldIsLeaf ? oldPred : 0.0;
    }
    
    n->makeLeaf(leafPrediction, leafPrediction);

    double msePruned = validate(n.get());
    
    // 还原子树状态计算原始误差
    n->isLeaf = oldIsLeaf;
    n->leftChild  = std::move(leftBackup);
    n->rightChild = std::move(rightBackup);
    
    if (!n->isLeaf) {
        // 恢复内部节点状态
        n->info.internal.left = n->leftChild.get();
        n->info.internal.right = n->rightChild.get();
    } else {
        // 恢复叶节点状态
        n->info.leaf.prediction = oldPred;
    }
    
    double mseOriginal = validate(n.get());

    // 如果剪枝后误差不增加（或减少），则保持剪枝
    if (msePruned <= mseOriginal) {
        // 执行剪枝
        n->makeLeaf(leafPrediction, leafPrediction);
    }
    // 否则保持原状态（已经还原了）
}

void ReducedErrorPruner::prune(std::unique_ptr<Node>& root) const {
    pruneRec(root);
}