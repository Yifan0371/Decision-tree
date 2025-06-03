// src/tree/criterion/MAECriterion.cpp - OpenMP并行版本
#include "criterion/MAECriterion.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

/* --------- 工具：并行子集中位数 ---------- */
static double subsetMedianParallel(const std::vector<double>& y,
                                   const std::vector<int>& idx)
{
    std::vector<double> v;
    v.reserve(idx.size());
    
    // 并行复制数据（如果数据量大的话）
    if (idx.size() > 1000) {
        v.resize(idx.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < idx.size(); ++i) {
            v[i] = y[idx[i]];
        }
    } else {
        // 小数据量直接串行复制
        for (int i : idx) {
            v.push_back(y[i]);
        }
    }

    const size_t n = v.size();
    auto mid_it = v.begin() + n / 2;
    std::nth_element(v.begin(), mid_it, v.end());

    if (n % 2 == 1) {
        return *mid_it;
    } else {
        auto left_it = std::max_element(v.begin(), mid_it);
        return 0.5 * (*left_it + *mid_it);
    }
}

/* --------- 并行MAE 计算 ---------- */
double MAECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const
{
    if (indices.empty()) return 0.0;

    const double med = subsetMedianParallel(labels, indices);
    double sumAbs = 0.0;
    
    // 并行计算绝对偏差之和
    #pragma omp parallel for reduction(+:sumAbs) schedule(static) if(indices.size() > 1000)
    for (size_t i = 0; i < indices.size(); ++i) {
        sumAbs += std::abs(labels[indices[i]] - med);
    }
    
    return sumAbs / indices.size();
}