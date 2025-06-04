#pragma once

#include "tree/ISplitCriterion.hpp"

class QuantileCriterion : public ISplitCriterion {
public:
    explicit QuantileCriterion(double tau = 0.5) : tau_(tau) {}
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>&   idx) const override;
private:
    double tau_;          
};
