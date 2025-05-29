#ifndef TREE_ISPLITCRITERION_HPP
#define TREE_ISPLITCRITERION_HPP

#include <vector>

class ISplitCriterion {
public:
    virtual ~ISplitCriterion() = default;

    /** 节点纯度度量（MSE/MAE/Huber） */
    virtual double nodeMetric(const std::vector<double>& labels,
                              const std::vector<int>& indices) const = 0;
};

#endif // TREE_ISPLITCRITERION_HPP
