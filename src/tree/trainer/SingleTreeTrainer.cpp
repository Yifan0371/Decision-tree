
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "tree/Node.hpp"
#include <numeric>
#include <cmath>
#include <iostream>

SingleTreeTrainer::SingleTreeTrainer(std::unique_ptr<ISplitFinder>   finder,
                                     std::unique_ptr<ISplitCriterion> criterion,
                                     std::unique_ptr<IPruner>        pruner,
                                     int maxDepth,
                                     int minSamplesLeaf)
    : maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf),
      finder_(std::move(finder)),
      criterion_(std::move(criterion)),
      pruner_(std::move(pruner)) {}

void SingleTreeTrainer::train(const std::vector<double>& data,
                              int rowLength,
                              const std::vector<double>& labels) {
    root_ = std::make_unique<Node>();
    std::vector<int> indices(labels.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    splitNode(root_.get(), data, rowLength, labels, indices, 0);
    pruner_->prune(root_);
    
    // 简化的统计信息输出
    int treeDepth = 0, leafCount = 0;
    calculateTreeStats(root_.get(), 0, treeDepth, leafCount);
    std::cout << "Tree: depth=" << treeDepth << ", leaves=" << leafCount;
}

void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    node->metric  = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();

    // 停止条件检查（无调试输出）
    if (depth >= maxDepth_ || 
        indices.size() < 2 * (size_t)minSamplesLeaf_) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        return;
    }

    // 寻找最佳分裂
    int   bestFeat;
    double bestThr, bestImp;
    std::tie(bestFeat, bestThr, bestImp) =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    // 检查分裂有效性
    if (bestFeat < 0 || bestImp <= 0) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        return;
    }

    // 设置分裂信息
    node->featureIndex = bestFeat;
    node->threshold    = bestThr;

    // 根据最佳分裂点划分数据
    std::vector<int> leftIdx, rightIdx;
    for (int idx : indices) {
        double v = data[idx * rowLength + bestFeat];
        if (v <= bestThr) leftIdx.push_back(idx);
        else              rightIdx.push_back(idx);
    }

    // 检查分裂后样本数约束
    if (leftIdx.size() < (size_t)minSamplesLeaf_ || 
        rightIdx.size() < (size_t)minSamplesLeaf_) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        return;
    }

    // 创建子节点并递归分裂
    node->left  = std::make_unique<Node>();
    node->right = std::make_unique<Node>();

    splitNode(node->left.get(),  data, rowLength, labels, leftIdx,  depth + 1);
    splitNode(node->right.get(), data, rowLength, labels, rightIdx, depth + 1);
}

double SingleTreeTrainer::predict(const double* sample,
                                  int /* rowLength */) const {
    const Node* cur = root_.get();
    while (!cur->isLeaf) {
        double v = sample[cur->featureIndex];
        cur = (v <= cur->threshold ? cur->left.get() : cur->right.get());
    }
    return cur->prediction;
}

void SingleTreeTrainer::evaluate(const std::vector<double>& X,
                                 int rowLength,
                                 const std::vector<double>& y,
                                 double& mse,
                                 double& mae) {
    size_t n = y.size();
    mse = 0; mae = 0;
    for (size_t i = 0; i < n; ++i) {
        double pred = predict(&X[i * rowLength], rowLength);
        double diff = y[i] - pred;
        mse += diff * diff;
        mae += std::abs(diff);
    }
    mse /= n;
    mae /= n;
}

void SingleTreeTrainer::calculateTreeStats(const Node* node, int currentDepth, 
                                           int& maxDepth, int& leafCount) const {
    if (!node) return;
    
    maxDepth = std::max(maxDepth, currentDepth);
    
    if (node->isLeaf) {
        leafCount++;
    } else {
        calculateTreeStats(node->left.get(), currentDepth + 1, maxDepth, leafCount);
        calculateTreeStats(node->right.get(), currentDepth + 1, maxDepth, leafCount);
    }
}