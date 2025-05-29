#include "criterion/MSECriterion.hpp"
#include <numeric>
#include <cmath>

double MSECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0;
    double sum = 0;
    for (int idx : indices) sum += labels[idx];
    double mean = sum / indices.size();

    double mse = 0;
    for (int idx : indices) {
        double d = labels[idx] - mean;
        mse += d * d;
    }
    return mse / indices.size();
}
