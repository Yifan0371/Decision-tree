// QuantileCriterion.cpp
#include "criterion/QuantileCriterion.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>  // 新增 OpenMP 头文件

double QuantileCriterion::nodeMetric(const std::vector<double>& y,
                                     const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    const size_t n = idx.size();
    // 1. 并行拷贝 y[idx] 到 vals
    std::vector<double> vals(n);
    #pragma omp parallel for
    for (size_t t = 0; t < n; ++t) {
        vals[t] = y[idx[t]];
    }

    // 2. 串行地求 τ-分位值
    const size_t k = static_cast<size_t>(tau_ * (n - 1));
    // 注意：std::nth_element 会就地修改 vals
    const double q = [&]() {
        std::nth_element(vals.begin(), vals.begin() + k, vals.end());
        return vals[k];
    }();

    // 3. 并行计算 pinball 损失
    double pinball = 0.0;
    #pragma omp parallel for reduction(+:pinball)
    for (size_t t = 0; t < n; ++t) {
        double v = vals[t];
        // 如果 v < q，用 (tau_ - 1)*(v - q)；否则用 tau_*(v - q)
        pinball += (v - q) * ( (v < q) ? (tau_ - 1.0) : tau_ );
    }

    return pinball / static_cast<double>(n);
}
