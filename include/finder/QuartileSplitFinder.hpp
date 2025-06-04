#pragma once

#include "tree/ISplitFinder.hpp"

class QuartileSplitFinder : public ISplitFinder {
public:
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;
};
