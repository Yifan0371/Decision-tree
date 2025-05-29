#ifndef TREE_ISPLITFINDER_HPP
#define TREE_ISPLITFINDER_HPP

#include <tuple>
#include <vector>
#include "Node.hpp"
#include "ISplitCriterion.hpp"

class ISplitFinder {
public:
    virtual ~ISplitFinder() = default;

    /**
     * 找到最佳划分
     * @return (bestFeature, threshold, impurityDecrease)
     */
    virtual std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& labels,
                  const std::vector<int>& indices,
                  double currentMetric,
                  const ISplitCriterion& criterion) const = 0;
};

#endif // TREE_ISPLITFINDER_HPP
