#include "criterion/PoissonCriterion.hpp"
#include <cmath>
#include <limits>

double PoissonCriterion::nodeMetric(const std::vector<double>& y,
                                    const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    double sum = 0.0;
    for (int i : idx) sum += y[i];
    const double mu = std::max(sum / idx.size(), 1e-12); // 避免 log0

    double loss = 0.0;
    for (int i : idx) {
        double yi = std::max(y[i], 1e-12);
        loss += mu - yi * std::log(mu);
    }
    return loss / idx.size();
}
