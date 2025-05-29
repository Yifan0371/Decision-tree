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
    : finder_(std::move(finder)),
      criterion_(std::move(criterion)),
      pruner_(std::move(pruner)),
      maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf) {}

void SingleTreeTrainer::train(const std::vector<double>& data,
                              int rowLength,
                              const std::vector<double>& labels) {
    std::cout << "=== Training Parameters ===" << std::endl;
    std::cout << "Max Depth: " << maxDepth_ << std::endl;
    std::cout << "Min Samples Leaf: " << minSamplesLeaf_ << std::endl;
    std::cout << "Dataset size: " << labels.size() << " samples, " 
              << rowLength << " features" << std::endl;
    
    root_ = std::make_unique<Node>();
    std::vector<int> indices(labels.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    splitNode(root_.get(), data, rowLength, labels, indices, 0);
    pruner_->prune(root_);
    
    // 添加树统计信息
    int treeDepth = 0, leafCount = 0;
    calculateTreeStats(root_.get(), 0, treeDepth, leafCount);
    std::cout << "=== Final Tree Statistics ===" << std::endl;
    std::cout << "Actual tree depth: " << treeDepth << std::endl;
    std::cout << "Number of leaf nodes: " << leafCount << std::endl;
    std::cout << "================================" << std::endl;
}

void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    node->metric  = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();

    // 详细的调试输出
    if (depth <= 3) {
        std::cout << "Depth " << depth << ": " << indices.size() 
                  << " samples, metric=" << node->metric << std::endl;
    }

    // 修正停止条件
    // 条件1: 达到最大深度
    if (depth >= maxDepth_) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        if (depth <= 5) {
            std::cout << "  -> Leaf (max depth reached), prediction=" 
                      << node->prediction << std::endl;
        }
        return;
    }

    // 条件2: 样本数太少，无法进行有意义的分裂
    // 需要至少 2*minSamplesLeaf 才能分裂（左右各至少 minSamplesLeaf）
    if (indices.size() < 2 * (size_t)minSamplesLeaf_) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        if (depth <= 5) {
            std::cout << "  -> Leaf (insufficient samples for split: " 
                      << indices.size() << " < " << (2 * minSamplesLeaf_) 
                      << "), prediction=" << node->prediction << std::endl;
        }
        return;
    }

    // 寻找最佳分裂
    int   bestFeat;
    double bestThr, bestImp;
    std::tie(bestFeat, bestThr, bestImp) =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    // 条件3: 找不到有效的分裂点
    if (bestFeat < 0 || bestImp <= 0) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        if (depth <= 5) {
            std::cout << "  -> Leaf (no beneficial split found), prediction=" 
                      << node->prediction << std::endl;
        }
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

    // 关键修复：检查分裂后是否满足 minSamplesLeaf 约束
    if (leftIdx.size() < (size_t)minSamplesLeaf_ || 
        rightIdx.size() < (size_t)minSamplesLeaf_) {
        node->isLeaf = true;
        double sum = 0;
        for (int idx : indices) sum += labels[idx];
        node->prediction = sum / indices.size();
        
        if (depth <= 5) {
            std::cout << "  -> Leaf (minSamplesLeaf constraint violated: left=" 
                      << leftIdx.size() << ", right=" << rightIdx.size() 
                      << ", required=" << minSamplesLeaf_ << "), prediction=" 
                      << node->prediction << std::endl;
        }
        return;
    }

    // 输出分裂信息
    if (depth <= 3) {
        std::cout << "  -> Split on feature " << bestFeat 
                  << " <= " << bestThr << " (improvement=" << bestImp 
                  << ", left=" << leftIdx.size() 
                  << ", right=" << rightIdx.size() << ")" << std::endl;
    }

    // 创建子节点并递归分裂
    node->left  = std::make_unique<Node>();
    node->right = std::make_unique<Node>();

    splitNode(node->left.get(),  data, rowLength, labels, leftIdx,  depth + 1);
    splitNode(node->right.get(), data, rowLength, labels, rightIdx, depth + 1);
}

double SingleTreeTrainer::predict(const double* sample,
                                  int rowLength) const {
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

// 计算树统计信息的辅助函数
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