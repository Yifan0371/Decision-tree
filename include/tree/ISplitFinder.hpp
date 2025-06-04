#pragma once

#include <tuple>
#include <vector>
#include "Node.hpp"
#include "ISplitCriterion.hpp"

class ISplitFinder {
public:
    virtual ~ISplitFinder() = default;

    
    virtual std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& labels,
                  const std::vector<int>& indices,
                  double currentMetric,
                  const ISplitCriterion& criterion) const = 0;
};
