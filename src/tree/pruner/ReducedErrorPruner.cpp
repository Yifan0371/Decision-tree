#include "pruner/ReducedErrorPruner.hpp"
#include <cmath>

double ReducedErrorPruner::validate(Node* n) const {
    double mse = 0.0;
    for (size_t i = 0; i < yv_.size(); ++i) {
        const double* sample = &Xv_[i * D_];
        Node* cur = n;
        while (!cur->isLeaf) {
            cur = (sample[cur->featureIndex] <= cur->threshold)
                    ? cur->left.get() : cur->right.get();
        }
        double diff = yv_[i] - cur->prediction;
        mse += diff * diff;
    }
    return mse / yv_.size();
}

void ReducedErrorPruner::pruneRec(std::unique_ptr<Node>& n) const {
    if (!n || n->isLeaf) return;
    pruneRec(n->left);
    pruneRec(n->right);
    
    // 备份
    auto leftBackup  = std::move(n->left);
    auto rightBackup = std::move(n->right);
    bool oldIsLeaf   = n->isLeaf;
    double oldPred   = n->prediction;

    // 尝试变叶
    n->isLeaf     = true;
    n->left.reset();
    n->right.reset();
    n->prediction = n->nodePrediction;    // 已存叶均值 or 中位数

    double msePruned = validate(n.get());
    
    // 还原到原子树状态计算原始误差
    n->isLeaf = oldIsLeaf;
    n->left   = std::move(leftBackup);
    n->right  = std::move(rightBackup);
    n->prediction = oldPred;
    
    double mseOriginal = validate(n.get());

    if (msePruned <= mseOriginal) {
        // 保持剪枝
        n->isLeaf = true;
        n->left.reset();
        n->right.reset();
        n->prediction = n->nodePrediction;
    }
    // 否则保持原状态（已经还原了）
}

void ReducedErrorPruner::prune(std::unique_ptr<Node>& root) const {
    pruneRec(root);
}