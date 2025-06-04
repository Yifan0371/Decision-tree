// =============================================================================
// src/tree/trainer/SingleTreeTrainer.cpp - 任务队列策略优化版本
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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

// **任务队列数据结构**
struct SplitTask {
    Node* node;
    std::vector<int> indices;
    int depth;
    
    SplitTask(Node* n, std::vector<int>&& idx, int d) 
        : node(n), indices(std::move(idx)), depth(d) {}
};

// **线程安全的任务队列**
class TaskQueue {
private:
    std::queue<std::unique_ptr<SplitTask>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> finished_{false};
    
public:
    void push(std::unique_ptr<SplitTask> task) {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
        condition_.notify_one();
    }
    
    std::unique_ptr<SplitTask> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !tasks_.empty() || finished_; });
        
        if (tasks_.empty()) return nullptr;
        
        auto task = std::move(tasks_.front());
        tasks_.pop();
        return task;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.empty();
    }
    
    void finish() {
        finished_ = true;
        condition_.notify_all();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }
};

SingleTreeTrainer::SingleTreeTrainer(std::unique_ptr<ISplitFinder> finder,
                                     std::unique_ptr<ISplitCriterion> criterion,
                                     std::unique_ptr<IPruner> pruner,
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
    
    root_ = std::make_unique<Node>();
    
    // **移除 omp_set_num_threads 调用，改用环境变量控制**
    int numThreads = 1;
    #ifdef _OPENMP
    numThreads = omp_get_max_threads();
    std::cout << "Using " << numThreads << " OpenMP threads (controlled by OMP_NUM_THREADS)" << std::endl;
    #endif
    
    // 预分配索引数组
    std::vector<int> rootIndices(labels.size());
    std::iota(rootIndices.begin(), rootIndices.end(), 0);
    
    // **教授建议的任务队列/线程池模式**
    const bool useTaskQueue = (labels.size() > 1000 && numThreads > 1);
    
    if (useTaskQueue) {
        std::cout << "Large dataset detected, using task queue strategy" << std::endl;
        buildTreeWithTaskQueue(data, rowLength, labels, std::move(rootIndices));
    } else {
        std::cout << "Small dataset, using optimized recursive strategy" << std::endl;
        splitNodeOptimized(root_.get(), data, rowLength, labels, rootIndices, 0);
    }
    
    auto splitEnd = std::chrono::high_resolution_clock::now();
    
    // 后剪枝
    auto pruneStart = std::chrono::high_resolution_clock::now();
    pruner_->prune(root_);
    auto pruneEnd = std::chrono::high_resolution_clock::now();
    
    auto trainEnd = std::chrono::high_resolution_clock::now();
    
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
}

// **新方法：任务队列驱动的树构建**
void SingleTreeTrainer::buildTreeWithTaskQueue(const std::vector<double>& data,
                                               int rowLength,
                                               const std::vector<double>& labels,
                                               std::vector<int>&& rootIndices) {
    
    TaskQueue taskQueue;
    std::atomic<int> activeWorkers{0};
    std::atomic<int> totalTasks{0};
    
    // 创建根任务
    auto rootTask = std::make_unique<SplitTask>(root_.get(), std::move(rootIndices), 0);
    taskQueue.push(std::move(rootTask));
    totalTasks++;
    
    const int numWorkers = std::min(omp_get_max_threads(), 8); // 限制最大线程数
    
    #pragma omp parallel num_threads(numWorkers)
    {
        const int threadId = omp_get_thread_num();
        
        while (true) {
            auto task = taskQueue.pop();
            if (!task) break; // 队列已完成
            
            activeWorkers++;
            
            // 处理当前任务
            processTask(data, rowLength, labels, std::move(task), taskQueue, totalTasks);
            
            activeWorkers--;
            
            // 检查是否所有工作完成
            if (activeWorkers == 0 && taskQueue.empty()) {
                taskQueue.finish();
                break;
            }
        }
    }
    
    std::cout << "Task queue processing completed. Total tasks: " << totalTasks.load() << std::endl;
}

// **任务处理方法**
void SingleTreeTrainer::processTask(const std::vector<double>& data,
                                    int rowLength,
                                    const std::vector<double>& labels,
                                    std::unique_ptr<SplitTask> task,
                                    TaskQueue& taskQueue,
                                    std::atomic<int>& totalTasks) {
    
    Node* node = task->node;
    const auto& indices = task->indices;
    const int depth = task->depth;
    
    if (indices.empty()) {
        node->makeLeaf(0.0);
        return;
    }
    
    node->metric = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();
    
    // **高效计算节点预测值**
    double sum = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        sum += labels[indices[i]];
    }
    const double nodePrediction = sum / indices.size();
    
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

    // **原地分割**
    std::vector<int> leftIndices, rightIndices;
    leftIndices.reserve(indices.size());
    rightIndices.reserve(indices.size());
    
    for (int idx : indices) {
        if (data[idx * rowLength + bestFeat] <= bestThr) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    if (leftIndices.size() < static_cast<size_t>(minSamplesLeaf_) || 
        rightIndices.size() < static_cast<size_t>(minSamplesLeaf_)) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // 创建子节点
    node->makeInternal(bestFeat, bestThr);
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // **关键：将子节点任务加入队列**
    if (!leftIndices.empty()) {
        auto leftTask = std::make_unique<SplitTask>(
            node->leftChild.get(), std::move(leftIndices), depth + 1);
        taskQueue.push(std::move(leftTask));
        totalTasks++;
    }
    
    if (!rightIndices.empty()) {
        auto rightTask = std::make_unique<SplitTask>(
            node->rightChild.get(), std::move(rightIndices), depth + 1);
        taskQueue.push(std::move(rightTask));
        totalTasks++;
    }
}

// **优化的节点分裂（保留用于小数据集）**
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
    
    node->metric = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();
    
    // **高效计算节点预测值**
    double sum = 0.0;
    const size_t numSamples = indices.size();
    
    // **教授建议：避免过小数据集的并行开销**
    if (numSamples > 1000) {
        #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(4)
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    } else {
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    }
    const double nodePrediction = sum / numSamples;
    
    // 停止条件检查
    if (depth >= maxDepth_ || 
        indices.size() < 2 * static_cast<size_t>(minSamplesLeaf_) ||
        indices.size() < 2) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // 寻找最佳分裂
    auto [bestFeat, bestThr, bestGain] =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    if (bestFeat < 0 || bestGain <= 0) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // 预剪枝检查
    if (auto* prePruner = dynamic_cast<const MinGainPrePruner*>(pruner_.get())) {
        if (bestGain < prePruner->minGain()) {
            node->makeLeaf(nodePrediction, nodePrediction);
            return;
        }
    }

    // **原地分割策略**
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

    // 创建子节点
    node->makeInternal(bestFeat, bestThr);
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // 创建子索引向量
    std::vector<int> leftIndices(indices.begin(), partitionPoint);
    std::vector<int> rightIndices(partitionPoint, indices.end());
    
    // **谨慎的并行递归（仅在前几层使用）**
    const bool useParallelRecursion = (depth <= 2) && 
                                     (indices.size() > 2000) &&
                                     (leftIndices.size() > 500 && rightIndices.size() > 500);
    
    if (useParallelRecursion) {
        #pragma omp parallel sections num_threads(2)
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
        splitNodeOptimized(node->leftChild.get(), data, rowLength, 
                          labels, leftIndices, depth + 1);
        splitNodeOptimized(node->rightChild.get(), data, rowLength, 
                          labels, rightIndices, depth + 1);
    }
}

double SingleTreeTrainer::predict(const double* sample, int /* rowLength */) const {
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
    const size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    // **并行预测和误差计算，使用 num_threads 子句**
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 256) num_threads(4) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double pred = predict(&X[i * rowLength], rowLength);
        const double diff = y[i] - pred;
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

// 兼容性方法
void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    std::vector<int> mutableIndices = indices;
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