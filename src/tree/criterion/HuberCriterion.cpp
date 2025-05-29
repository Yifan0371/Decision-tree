#include "criterion/HuberCriterion.hpp"
#include <cmath>
#include <numeric>

double HuberCriterion::nodeMetric(const std::vector<double>& y,
                                  const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    /* 用均值做估计 */
    double sum = 0.0;
    for (int i : idx) sum += y[i];
    const double mu = sum / idx.size();

    double loss = 0.0;
    const double d = delta_;
    for (int i : idx) {
        double r = y[i] - mu;
        if (std::abs(r) <= d)
            loss += 0.5 * r * r;
        else
            loss += d * (std::abs(r) - 0.5 * d);
    }
    return loss / idx.size();
}
