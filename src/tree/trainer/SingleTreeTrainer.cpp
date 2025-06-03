// src/tree/trainer/SingleTreeTrainer.cpp - OpenMP并行版本
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "tree/Node.hpp"
#include "pruner/MinGainPrePruner.hpp"
#include <numeric>
#include <cmath>
#include <iostream>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

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
    
    // 并行初始化索引数组
    #pragma omp parallel for schedule(static) if(labels.size() > 10000)
    for (size_t i = 0; i < labels.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    
    // 使用并行的递归分裂
    splitNodeInPlaceParallel(root_.get(), data, rowLength, labels, indices, 0);
    
    // 训练完成后调用后剪枝
    pruner_->prune(root_);
    
    // 简化的统计信息输出
    int treeDepth = 0, leafCount = 0;
    calculateTreeStats(root_.get(), 0, treeDepth, leafCount);
    std::cout << "Tree: depth=" << treeDepth << ", leaves=" << leafCount;
}

void SingleTreeTrainer::splitNodeInPlaceParallel(Node* node,
                                                 const std::vector<double>& data,
                                                 int rowLength,
                                                 const std::vector<double>& labels,
                                                 std::vector<int>& indices,
                                                 int depth) {
    if (indices.empty()) {
        node->makeLeaf(0.0);
        return;
    }
    
    node->metric  = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();
    
    // 并行计算节点预测值（均值）
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) if(indices.size() > 1000)
    for (size_t i = 0; i < indices.size(); ++i) {
        sum += labels[indices[i]];
    }
    double nodePrediction = sum / indices.size();
    
    // 停止条件检查
    if (depth >= maxDepth_ || 
        indices.size() < 2 * static_cast<size_t>(minSamplesLeaf_) ||
        indices.size() < 2) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // 寻找最佳分裂
    int   bestFeat;
    double bestThr, bestGain;
    std::tie(bestFeat, bestThr, bestGain) =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    // 检查分裂有效性
    if (bestFeat < 0 || bestGain <= 0) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **预剪枝检查**：如果是 MinGainPrePruner，检查增益阈值
    if (auto* prePruner = dynamic_cast<const MinGainPrePruner*>(pruner_.get())) {
        if (bestGain < prePruner->minGain()) {
            node->makeLeaf(nodePrediction, nodePrediction);
            return;
        }
    }

    // **并行优化：使用std::partition进行就地划分**
    auto partitionPoint = std::partition(indices.begin(), indices.end(),
        [&](int idx) {
            return data[idx * rowLength + bestFeat] <= bestThr;
        });
    
    // 计算左右子集大小
    size_t leftSize = std::distance(indices.begin(), partitionPoint);
    size_t rightSize = indices.size() - leftSize;
    
    // 检查分裂后样本数约束
    if (leftSize < static_cast<size_t>(minSamplesLeaf_) || 
        rightSize < static_cast<size_t>(minSamplesLeaf_)) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // 设置为内部节点
    node->makeInternal(bestFeat, bestThr);
    
    // 创建子节点
    node->leftChild  = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    
    // 更新union中的指针（用于访问）
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // **内存优化：创建左右子集索引**
    std::vector<int> leftIndices(indices.begin(), partitionPoint);
    std::vector<int> rightIndices(partitionPoint, indices.end());
    
    // **并行递归处理：在合适的深度开始并行**
    // 只在树的上层启用并行（避免过度并行化的开销）
    const int PARALLEL_THRESHOLD_DEPTH = 3;
    const size_t PARALLEL_THRESHOLD_SIZE = 5000;
    
    bool useParallel = (depth < PARALLEL_THRESHOLD_DEPTH) && 
                      (leftIndices.size() > PARALLEL_THRESHOLD_SIZE || 
                       rightIndices.size() > PARALLEL_THRESHOLD_SIZE);
    
    if (useParallel) {
        // 使用OpenMP task并行处理左右子树
        #pragma omp parallel sections if(depth <= 2)
        {
            #pragma omp section
            {
                splitNodeInPlaceParallel(node->leftChild.get(), data, rowLength, 
                                        labels, leftIndices, depth + 1);
            }
            #pragma omp section  
            {
                splitNodeInPlaceParallel(node->rightChild.get(), data, rowLength, 
                                        labels, rightIndices, depth + 1);
            }
        }
    } else {
        // 串行递归处理（避免过度并行化）
        splitNodeInPlaceParallel(node->leftChild.get(),  data, rowLength, 
                                labels, leftIndices,  depth + 1);
        splitNodeInPlaceParallel(node->rightChild.get(), data, rowLength, 
                                labels, rightIndices, depth + 1);
    }
}

double SingleTreeTrainer::predict(const double* sample,
                                  int /* rowLength */) const {
    const Node* cur = root_.get();
    while (!cur->isLeaf) {
        double v = sample[cur->getFeatureIndex()];
        cur = (v <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur->getPrediction();
}

void SingleTreeTrainer::evaluate(const std::vector<double>& X,
                                 int rowLength,
                                 const std::vector<double>& y,
                                 double& mse,
                                 double& mae) {
    size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    // 并行计算预测和误差
    #pragma omp parallel for reduction(+:mse,mae) schedule(static) if(n > 1000)
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
        calculateTreeStats(node->getLeft(), currentDepth + 1, maxDepth, leafCount);
        calculateTreeStats(node->getRight(), currentDepth + 1, maxDepth, leafCount);
    }
}

// 保留原有splitNode方法以维持API兼容性
void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    // 创建索引的副本以供就地修改
    std::vector<int> mutableIndices = indices;
    splitNodeInPlaceParallel(node, data, rowLength, labels, mutableIndices, depth);
}

// 新增并行版本的splitNodeInPlace，与原版API兼容
void SingleTreeTrainer::splitNodeInPlace(Node* node,
                                         const std::vector<double>& data,
                                         int rowLength,
                                         const std::vector<double>& labels,
                                         std::vector<int>& indices,
                                         int depth) {
    splitNodeInPlaceParallel(node, data, rowLength, labels, indices, depth);
}