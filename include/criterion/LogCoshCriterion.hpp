#ifndef LOGCOSH_CRITERION_HPP
#define LOGCOSH_CRITERION_HPP

#include "tree/ISplitCriterion.hpp"

class LogCoshCriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>&   idx) const override;
};

#endif
