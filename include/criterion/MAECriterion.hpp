#pragma once

#include "tree/ISplitCriterion.hpp"
#include <vector>


class MAECriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override;
};
