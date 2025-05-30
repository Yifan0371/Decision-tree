// src/tree/criterion/MSECriterion.cpp
#include "criterion/MSECriterion.hpp"
#include <numeric>
#include <cmath>
#include <immintrin.h> // for SIMD

// 启用编译器自动向量化
#pragma GCC optimize("O3")
#pragma GCC target("avx2")

double MSECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0;
    
    const size_t n = indices.size();
    
    // 对于小数据集，使用简单计算
    if (n <= 4) {
        double sum = 0.0;
        for (int idx : indices) {
            sum += labels[idx];
        }
        double mean = sum / n;
        
        double mse = 0.0;
        for (int idx : indices) {
            double d = labels[idx] - mean;
            mse += d * d;
        }
        return mse / n;
    }
    
    // 对于大数据集，使用向量化计算
    MetricCache cache;
    calculateStats(labels, indices, cache);
    return cache.mse;
}

void MSECriterion::calculateStats(const std::vector<double>& labels,
                                 const std::vector<int>& indices,
                                 MetricCache& cache) {
    if (indices.empty()) {
        cache = {};
        return;
    }
    
    const size_t n = indices.size();
    cache.count = n;
    
    // 使用 Kahan 求和算法提高数值稳定性
    double sum = 0.0, c = 0.0;
    
    // 尽可能让编译器向量化
    const int* idx_ptr = indices.data();
    
    #pragma GCC ivdep  // 告诉编译器没有数据依赖
    for (size_t i = 0; i < n; ++i) {
        double y = labels[idx_ptr[i]] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    cache.sum = sum;
    cache.mean = sum / n;
    
    // 计算平方和，使用向量化友好的方式
    double sumSq = 0.0;
    c = 0.0;
    
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        double diff = labels[idx_ptr[i]] - cache.mean;
        double y = diff * diff - c;
        double t = sumSq + y;
        c = (t - sumSq) - y;
        sumSq = t;
    }
    
    cache.sumSq = sumSq;
    cache.mse = sumSq / n;
    cache.valid = true;
}

double MSECriterion::splitMetric(const MetricCache& leftCache,
                                const MetricCache& rightCache) {
    if (!leftCache.valid || !rightCache.valid) return 0.0;
    
    size_t totalCount = leftCache.count + rightCache.count;
    if (totalCount == 0) return 0.0;
    
    // 加权平均 MSE
    double weightedMSE = (leftCache.mse * leftCache.count + 
                         rightCache.mse * rightCache.count) / totalCount;
    
    return weightedMSE;
}