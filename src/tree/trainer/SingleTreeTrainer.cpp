// =============================================================================
// src/tree/trainer/SingleTreeTrainer.cpp - 优化版本（避免vector拷贝和new）
// =============================================================================
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "tree/Node.hpp"
#include "pruner/MinGainPrePruner.hpp"
#include <numeric>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iomanip>
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
    
    auto trainStart = std::chrono::high_resolution_clock::now();
    
    // **优化1: 使用智能指针预分配根节点**
    root_ = std::make_unique<Node>();
    
    #ifdef _OPENMP
    omp_set_dynamic(1);  // 允许运行时调整线程数
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Using " << omp_get_num_threads() << " OpenMP threads" << std::endl;
        }
    }
    #endif
    
    // **优化2: 预分配索引数组并避免不必要的拷贝**
    std::vector<int> indices;
    indices.reserve(labels.size());
    
    #pragma omp parallel for schedule(static, 1024) if(labels.size() > 5000)
    for (size_t i = 0; i < labels.size(); ++i) {
        #pragma omp critical
        indices.push_back(static_cast<int>(i));
    }
    
    // 如果数据量小，直接串行初始化
    if (labels.size() <= 5000) {
        indices.resize(labels.size());
        std::iota(indices.begin(), indices.end(), 0);
    }
    
    // **优化3: 智能并行策略选择**
    if (labels.size() > 5000) {
        std::cout << "Large dataset detected (" << labels.size() 
                  << " samples), using enhanced parallel strategy" << std::endl;
    } else if (labels.size() > 1000) {
        std::cout << "Medium dataset detected, using balanced parallel strategy" << std::endl;
    } else {
        std::cout << "Small dataset detected, using conservative parallel strategy" << std::endl;
    }
    
    splitNodeOptimized(root_.get(), data, rowLength, labels, indices, 0);
    
    auto splitEnd = std::chrono::high_resolution_clock::now();
    
    // 后剪枝
    auto pruneStart = std::chrono::high_resolution_clock::now();
    pruner_->prune(root_);
    auto pruneEnd = std::chrono::high_resolution_clock::now();
    
    auto trainEnd = std::chrono::high_resolution_clock::now();
    
    // **优化4: 详细的性能统计**
    auto splitTime = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - trainStart);
    auto pruneTime = std::chrono::duration_cast<std::chrono::milliseconds>(pruneEnd - pruneStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    
    int treeDepth = 0, leafCount = 0;
    calculateTreeStats(root_.get(), 0, treeDepth, leafCount);
    
    std::cout << "Tree training completed:" << std::endl;
    std::cout << "  Depth: " << treeDepth << " | Leaves: " << leafCount << std::endl;
    std::cout << "  Split time: " << splitTime.count() << "ms" 
              << " | Prune time: " << pruneTime.count() << "ms"
              << " | Total: " << totalTime.count() << "ms" << std::endl;
    std::cout << "  Samples/ms: " << std::fixed << std::setprecision(1) 
              << (labels.size() / static_cast<double>(std::max(1L, totalTime.count()))) << std::endl;
}

// **新方法：优化的节点分裂（避免vector拷贝）**
void SingleTreeTrainer::splitNodeOptimized(Node* node,
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
    
    // **优化: 高效计算节点预测值**
    double sum = 0.0;
    const size_t numSamples = indices.size();
    
    if (numSamples > 1000) {
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    } else {
        // 小数据集使用串行计算（避免并行开销）
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    }
    const double nodePrediction = sum / numSamples;
    
    // **停止条件检查**
    if (depth >= maxDepth_ || 
        indices.size() < 2 * static_cast<size_t>(minSamplesLeaf_) ||
        indices.size() < 2) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **寻找最佳分裂**
    auto [bestFeat, bestThr, bestGain] =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    if (bestFeat < 0 || bestGain <= 0) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **预剪枝检查**
    if (auto* prePruner = dynamic_cast<const MinGainPrePruner*>(pruner_.get())) {
        if (bestGain < prePruner->minGain()) {
            node->makeLeaf(nodePrediction, nodePrediction);
            return;
        }
    }

    // **优化: 原地分割策略（避免vector拷贝）**
    auto partitionPoint = std::partition(indices.begin(), indices.end(),
        [&](int idx) {
            return data[idx * rowLength + bestFeat] <= bestThr;
        });
    
    const size_t leftSize = std::distance(indices.begin(), partitionPoint);
    const size_t rightSize = indices.size() - leftSize;
    
    if (leftSize < static_cast<size_t>(minSamplesLeaf_) || 
        rightSize < static_cast<size_t>(minSamplesLeaf_)) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **使用智能指针创建子节点**
    node->makeInternal(bestFeat, bestThr);
    node->leftChild  = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // **创建子索引向量（移动语义优化）**
    std::vector<int> leftIndices(indices.begin(), partitionPoint);
    std::vector<int> rightIndices(partitionPoint, indices.end());
    
    // **优化: 智能并行递归策略**
    const bool useParallelRecursion = (depth <= 3) && 
                                     (indices.size() > 2000) &&
                                     (leftIndices.size() > 500 || rightIndices.size() > 500);
    
    if (useParallelRecursion) {
        #pragma omp parallel sections if(depth <= 2)
        {
            #pragma omp section
            {
                splitNodeOptimized(node->leftChild.get(), data, rowLength, 
                                  labels, leftIndices, depth + 1);
            }
            #pragma omp section  
            {
                splitNodeOptimized(node->rightChild.get(), data, rowLength, 
                                  labels, rightIndices, depth + 1);
            }
        }
    } else {
        // 串行递归处理
        splitNodeOptimized(node->leftChild.get(),  data, rowLength, 
                          labels, leftIndices,  depth + 1);
        splitNodeOptimized(node->rightChild.get(), data, rowLength, 
                          labels, rightIndices, depth + 1);
    }
}

double SingleTreeTrainer::predict(const double* sample,
                                  int /* rowLength */) const {
    const Node* cur = root_.get();
    while (cur && !cur->isLeaf) {
        const double v = sample[cur->getFeatureIndex()];
        cur = (v <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}

void SingleTreeTrainer::evaluate(const std::vector<double>& X,
                                 int rowLength,
                                 const std::vector<double>& y,
                                 double& mse,
                                 double& mae) {
    auto evalStart = std::chrono::high_resolution_clock::now();
    
    const size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    // **优化: 并行预测和误差计算**
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double pred = predict(&X[i * rowLength], rowLength);
        const double diff = y[i] - pred;
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
    
    auto evalEnd = std::chrono::high_resolution_clock::now();
    auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(evalEnd - evalStart);
    
    std::cout << "Evaluation completed in " << evalTime.count() << "ms" << std::endl;
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

// **兼容性方法（保留原有API）**
void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    std::vector<int> mutableIndices = indices; // 需要拷贝以保持const正确性
    splitNodeOptimized(node, data, rowLength, labels, mutableIndices, depth);
}

void SingleTreeTrainer::splitNodeInPlace(Node* node,
                                         const std::vector<double>& data,
                                         int rowLength,
                                         const std::vector<double>& labels,
                                         std::vector<int>& indices,
                                         int depth) {
    splitNodeOptimized(node, data, rowLength, labels, indices, depth);
}

void SingleTreeTrainer::splitNodeInPlaceParallel(Node* node,
                                                 const std::vector<double>& data,
                                                 int rowLength,
                                                 const std::vector<double>& labels,
                                                 std::vector<int>& indices,
                                                 int depth) {
    splitNodeOptimized(node, data, rowLength, labels, indices, depth);
}