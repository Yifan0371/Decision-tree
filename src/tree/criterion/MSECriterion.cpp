// src/tree/criterion/MSECriterion.cpp - OpenMP并行版本
#include "criterion/MSECriterion.hpp"
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

double MSECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0;
    
    size_t n = indices.size();
    
    // **并行优化：使用OpenMP并行reduction计算和与平方和**
    double sum = 0.0;
    double sumSq = 0.0;
    
    #pragma omp parallel for reduction(+:sum,sumSq) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        double y = labels[indices[i]];
        sum += y;
        sumSq += y * y;
    }
    
    // 使用公式计算MSE：MSE = E[y²] - (E[y])²
    double mean = sum / n;
    double mse = sumSq / n - mean * mean;
    
    // 处理数值精度问题，确保非负
    return std::max(0.0, mse);
}