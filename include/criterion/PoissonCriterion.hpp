#pragma once

#include "tree/ISplitCriterion.hpp"

class PoissonCriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>&   idx) const override;
};
