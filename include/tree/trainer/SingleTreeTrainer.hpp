#pragma once

#include "../ITreeTrainer.hpp"
#include "../ISplitFinder.hpp"
#include "../ISplitCriterion.hpp"
#include "../IPruner.hpp"
#include "../../pruner/MinGainPrePruner.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <queue>
#include <atomic>

// 前向声明
struct SplitTask;
class TaskQueue;

class SingleTreeTrainer : public ITreeTrainer {
public:
    SingleTreeTrainer(std::unique_ptr<ISplitFinder>   finder,
                      std::unique_ptr<ISplitCriterion> criterion,
                      std::unique_ptr<IPruner>         pruner,
                      int maxDepth,
                      int minSamplesLeaf);

    void train(const std::vector<double>& data,
               int rowLength,
               const std::vector<double>& labels) override;

    double predict(const double* sample,
                   int rowLength) const override;

    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae) override;

private:
    // **新增：任务队列驱动的树构建方法**
    void buildTreeWithTaskQueue(const std::vector<double>& data,
                                int rowLength,
                                const std::vector<double>& labels,
                                std::vector<int>&& rootIndices);
    
    void processTask(const std::vector<double>& data,
                     int rowLength,
                     const std::vector<double>& labels,
                     std::unique_ptr<SplitTask> task,
                     TaskQueue& taskQueue,
                     std::atomic<int>& totalTasks);
    
    // 原有方法（优化版本）
    void splitNode(Node* node,
                   const std::vector<double>& data,
                   int rowLength,
                   const std::vector<double>& labels,
                   const std::vector<int>& indices,
                   int depth);

    void splitNodeInPlace(Node* node,
                          const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels,
                          std::vector<int>& indices,
                          int depth);

    void splitNodeInPlaceParallel(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  std::vector<int>& indices,
                                  int depth);

    void calculateTreeStats(const Node* node,
                            int currentDepth,
                            int& maxDepth,
                            int& leafCount) const;
                            
    void splitNodeOptimized(Node* node,
                           const std::vector<double>& data,
                           int rowLength,
                           const std::vector<double>& labels,
                           std::vector<int>& indices,
                           int depth);

    int  maxDepth_;
    int  minSamplesLeaf_;
    std::unique_ptr<ISplitFinder>    finder_;
    std::unique_ptr<ISplitCriterion> criterion_;
    std::unique_ptr<IPruner>         pruner_;
    
    // **教授建议：友元类允许 BaggingTrainer 访问内部结构**
    friend class BaggingTrainer;
    friend class IndexedSingleTreeTrainer;
};