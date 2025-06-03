// include/tree/trainer/SingleTreeTrainer.hpp - OpenMP并行版本
#ifndef TREE_SINGLE_TRAINER_HPP
#define TREE_SINGLE_TRAINER_HPP

#include "../ITreeTrainer.hpp"
#include "../ISplitFinder.hpp"
#include "../ISplitCriterion.hpp"
#include "../IPruner.hpp"
#include "../../pruner/MinGainPrePruner.hpp"  // for dynamic_cast
#include <memory>
#include <vector>
#include <iostream>

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
    // 原有方法（保持API兼容性）
    void splitNode(Node* node,
                   const std::vector<double>& data,
                   int rowLength,
                   const std::vector<double>& labels,
                   const std::vector<int>& indices,
                   int depth);

    // **原有的就地划分方法**
    void splitNodeInPlace(Node* node,
                          const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels,
                          std::vector<int>& indices,  // 非const引用，允许就地修改
                          int depth);

    // **新增：并行版本的就地划分方法**
    void splitNodeInPlaceParallel(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  std::vector<int>& indices,  // 非const引用，允许就地修改
                                  int depth);

    void calculateTreeStats(const Node* node,
                            int currentDepth,
                            int& maxDepth,
                            int& leafCount) const;

    int  maxDepth_;
    int  minSamplesLeaf_;
    std::unique_ptr<ISplitFinder>    finder_;
    std::unique_ptr<ISplitCriterion> criterion_;
    std::unique_ptr<IPruner>         pruner_;
    // 移除重复的 root_ 定义，使用基类的 root_
};

#endif // TREE_SINGLE_TRAINER_HPP