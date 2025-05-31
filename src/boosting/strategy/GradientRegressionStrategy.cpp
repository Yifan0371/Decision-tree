
// =============================================================================
// src/boosting/strategy/GradientRegressionStrategy.cpp  
// =============================================================================
#include "boosting/strategy/GradientRegressionStrategy.hpp"
#include <algorithm>
#include <cmath>

double GradientRegressionStrategy::computeOptimalLearningRate(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const std::vector<double>& tree_pred) const {
    
    // 简单的黄金分割线搜索
    double low = 0.0, high = 1.0;
    const double phi = 0.618033988749;
    const int maxIter = 10;  // 减少迭代次数
    const double tol = 1e-3;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        double mid1 = low + (1 - phi) * (high - low);
        double mid2 = low + phi * (high - low);
        
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
    
    double totalLoss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double newPred = y_pred[i] + lr * tree_pred[i];
        totalLoss += lossFunc_->loss(y_true[i], newPred);
    }
    return totalLoss / y_true.size();
}