#ifndef EXHAUSTIVE_SPLIT_FINDER_HPP
#define EXHAUSTIVE_SPLIT_FINDER_HPP

#include "../tree/ISplitFinder.hpp"

class ExhaustiveSplitFinder : public ISplitFinder {
public:
    std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& labels,
                  const std::vector<int>& indices,
                  double currentMetric,
                  const ISplitCriterion& criterion) const override;
};

#endif // EXHAUSTIVE_SPLIT_FINDER_HPP
