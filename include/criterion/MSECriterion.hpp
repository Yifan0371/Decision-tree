#ifndef MSE_CRITERION_HPP
#define MSE_CRITERION_HPP

#include "../tree/ISplitCriterion.hpp"

class MSECriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override;
};

#endif // MSE_CRITERION_HPP
