// =============================================================================
// src/boosting/loss/IRegressionLoss.cpp - OpenMP并行优化版本
// =============================================================================
#include "boosting/loss/IRegressionLoss.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void IRegressionLoss::computeGradientsHessians(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients,
    std::vector<double>& hessians) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    hessians.resize(n);
    
    // **并行优化1: 批量梯度和Hessian计算的并行**
    // 这是XGBoost和其他二阶优化算法的性能关键路径
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = gradient(y_true[i], y_pred[i]);
        hessians[i] = hessian(y_true[i], y_pred[i]);
    }
}

// =============================================
// 新增：批量损失计算的并行版本
// =============================================

double IRegressionLoss::computeBatchLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **并行优化2: 批量损失计算的并行reduction**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += loss(y_true[i], y_pred[i]);
    }
    
    return totalLoss / n;
}

void IRegressionLoss::computeBatchGradients(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    
    // **并行优化3: 仅梯度计算的并行（GBDT常用）**
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = gradient(y_true[i], y_pred[i]);
    }
}

// =============================================
// 新增：SIMD友好的向量化版本（编译器自动向量化）
// =============================================

void IRegressionLoss::computeGradientsVectorized(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    
    // **并行优化4: SIMD友好的向量化版本**
    // 使用更大的块大小，便于编译器进行向量化优化
    #pragma omp parallel for schedule(static, 2048) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        // 简化的内存访问模式，便于SIMD优化
        double yt = y_true[i];
        double yp = y_pred[i];
        gradients[i] = gradient(yt, yp);
    }
}

// =============================================
// 新增：性能监控版本
// =============================================

double IRegressionLoss::computeBatchLossWithTiming(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    double& computeTimeMs) const {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = computeBatchLoss(y_true, y_pred);
    
    auto end = std::chrono::high_resolution_clock::now();
    computeTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    return result;
}