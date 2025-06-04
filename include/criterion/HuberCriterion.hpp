#pragma once

#include "tree/ISplitCriterion.hpp"

class HuberCriterion : public ISplitCriterion {
public:
    explicit HuberCriterion(double delta = 1.0) : delta_(delta) {}
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>&   idx) const override;
private:
    double delta_;
};
