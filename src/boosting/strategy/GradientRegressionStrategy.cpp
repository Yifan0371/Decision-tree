// =============================================================================
// src/boosting/strategy/GradientRegressionStrategy.cpp - OpenMP并行优化版本
// =============================================================================
#include "boosting/strategy/GradientRegressionStrategy.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

void GradientRegressionStrategy::updateTargets(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& targets) const {
    
    size_t n = y_true.size();
    targets.resize(n);
    
    // **并行优化1: 残差/梯度计算的并行**
    // 每个样本的梯度计算完全独立，是理想的并行场景
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        targets[i] = lossFunc_->gradient(y_true[i], y_pred[i]);
    }
}

void GradientRegressionStrategy::updatePredictions(
    const std::vector<double>& tree_pred,
    double learning_rate,
    std::vector<double>& y_pred) const {
    
    size_t n = y_pred.size();
    
    // **并行优化2: 预测更新的并行**
    // 每个样本的预测更新也是完全独立的
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        y_pred[i] += learning_rate * tree_pred[i];
    }
}

double GradientRegressionStrategy::computeTotalLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **并行优化3: 损失计算的并行reduction**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += lossFunc_->loss(y_true[i], y_pred[i]);
    }
    
    return totalLoss / n;
}

double GradientRegressionStrategy::computeOptimalLearningRate(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const std::vector<double>& tree_pred) const {
    
    // 简单的黄金分割线搜索（并行优化版本）
    double low = 0.0, high = 1.0;
    const double phi = 0.618033988749;
    const int maxIter = 10;
    const double tol = 1e-3;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        double mid1 = low + (1 - phi) * (high - low);
        double mid2 = low + phi * (high - low);
        
        // **并行优化4: 线搜索中的损失评估并行化**
        double loss1 = evaluateLoss(y_true, y_pred, tree_pred, mid1);
        double loss2 = evaluateLoss(y_true, y_pred, tree_pred, mid2);
        
        if (loss1 < loss2) {
            high = mid2;
        } else {
            low = mid1;
        }
        
        if (std::abs(high - low) < tol) break;
    }
    
    return (low + high) * 0.5;
}

double GradientRegressionStrategy::evaluateLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const std::vector<double>& tree_pred,
    double lr) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **并行优化5: 线搜索损失评估的并行**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        double newPred = y_pred[i] + lr * tree_pred[i];
        totalLoss += lossFunc_->loss(y_true[i], newPred);
    }
    
    return totalLoss / n;
}