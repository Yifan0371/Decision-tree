// PoissonCriterion.cpp
#include "criterion/PoissonCriterion.hpp"
#include <cmath>
#include <limits>
#include <omp.h>  // 新增 OpenMP 头文件

double PoissonCriterion::nodeMetric(const std::vector<double>& y,
                                    const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    const int n = static_cast<int>(idx.size());
    // 1. 并行计算 sum，用于后续计算 mu（防止 log(0)）
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int t = 0; t < n; ++t) {
        sum += y[idx[t]];
    }
    // 若均值为 0，则用极小值避免 log(0)
    const double mu = std::max(sum / n, 1e-12);

    // 2. 并行计算 Poisson 损失
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss)
    for (int t = 0; t < n; ++t) {
        // 避免 yi 为 0 导致 log 问题
        double yi = std::max(y[idx[t]], 1e-12);
        loss += mu - yi * std::log(mu);
    }

    return loss / n;
}
