#include "criterion/QuantileCriterion.hpp"
#include <algorithm>
#include <cmath>

static double kth_value(std::vector<double>& v, size_t k)
{
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

double QuantileCriterion::nodeMetric(const std::vector<double>& y,
                                     const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    std::vector<double> vals;
    vals.reserve(idx.size());
    for (int i : idx) vals.push_back(y[i]);

    const size_t n = vals.size();
    const size_t k = static_cast<size_t>(tau_ * (n - 1));
    const double q = kth_value(vals, k);          // τ-分位

    double pinball = 0.0;
    for (double v : vals)
        pinball += (v - q) * ( (v < q) ? (tau_ - 1.0) : tau_);

    return pinball / n;
}
