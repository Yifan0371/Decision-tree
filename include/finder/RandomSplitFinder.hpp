#pragma once

#include "tree/ISplitFinder.hpp"
#include <random>
#include <tuple>
#include <vector>

class RandomSplitFinder : public ISplitFinder {
public:
    explicit RandomSplitFinder(int k = 10, uint32_t seed = 42)
      : k_(k), gen_(seed) {}
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;
private:
    int               k_;
    mutable std::mt19937 gen_;  
};
