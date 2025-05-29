#include "criterion/LogCoshCriterion.hpp"
#include <cmath>
#include <numeric>

double LogCoshCriterion::nodeMetric(const std::vector<double>& y,
                                    const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    double sum = 0.0;
    for (int i : idx) sum += y[i];
    const double mu = sum / idx.size();

    double loss = 0.0;
    for (int i : idx)
        loss += std::log(std::cosh(y[i] - mu));

    return loss / idx.size();
}
