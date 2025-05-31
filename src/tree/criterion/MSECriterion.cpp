#include "criterion/MSECriterion.hpp"
#include <numeric>
#include <cmath>

double MSECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0;
    
    size_t n = indices.size();
    
    // **内存优化：一次遍历同时计算和与平方和**
    double sum = 0.0;
    double sumSq = 0.0;
    
    for (int idx : indices) {
        double y = labels[idx];
        sum += y;
        sumSq += y * y;
    }
    
    // 使用公式计算MSE：MSE = E[y²] - (E[y])²
    double mean = sum / n;
    double mse = sumSq / n - mean * mean;
    
    // 处理数值精度问题，确保非负
    return std::max(0.0, mse);
}