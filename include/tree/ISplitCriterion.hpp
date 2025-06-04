#pragma once

#include <vector>

class ISplitCriterion {
public:
    virtual ~ISplitCriterion() = default;

    
    virtual double nodeMetric(const std::vector<double>& labels,
                              const std::vector<int>& indices) const = 0;
};
