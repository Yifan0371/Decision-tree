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
    void splitNode(Node* node,
                   const std::vector<double>& data,
                   int rowLength,
                   const std::vector<double>& labels,
                   const std::vector<int>& indices,
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
    std::unique_ptr<Node>            root_;
};

#endif // TREE_SINGLE_TRAINER_HPP
