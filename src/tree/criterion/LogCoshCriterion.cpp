// LogCoshCriterion.cpp
#include "criterion/LogCoshCriterion.hpp"
#include <cmath>
#include <numeric>
#include <omp.h>  // 新增 OpenMP 头文件

double LogCoshCriterion::nodeMetric(const std::vector<double>& y,
                                    const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    const int n = static_cast<int>(idx.size());
    // 1. 并行计算 sum 用于计算均值 mu
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int t = 0; t < n; ++t) {
        sum += y[idx[t]];
    }
    const double mu = sum / n;

    // 2. 并行计算 log(cosh) 损失
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss)
    for (int t = 0; t < n; ++t) {
        double r = y[idx[t]] - mu;
        loss += std::log(std::cosh(r));
    }

    return loss / n;
}
