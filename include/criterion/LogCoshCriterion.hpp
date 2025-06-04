#pragma once

#include "tree/ISplitCriterion.hpp"

class LogCoshCriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>&   idx) const override;
};
