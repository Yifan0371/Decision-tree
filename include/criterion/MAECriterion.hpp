#ifndef MAE_CRITERION_HPP
#define MAE_CRITERION_HPP

#include "tree/ISplitCriterion.hpp"
#include <vector>

/**
 *  Mean Absolute Error (MAE) 基准：
 *      metric = 1/n · Σ |y_i − median|
 *  相较 MSE 对离群点鲁棒。
 */
class MAECriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override;
};

#endif // MAE_CRITERION_HPP
