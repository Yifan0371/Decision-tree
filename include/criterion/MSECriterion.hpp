#pragma once

#include "../tree/ISplitCriterion.hpp"

class MSECriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override;
};
