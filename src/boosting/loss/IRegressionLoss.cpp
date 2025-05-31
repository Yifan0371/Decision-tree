// =============================================================================
// src/boosting/loss/IRegressionLoss.cpp
// =============================================================================
#include "boosting/loss/IRegressionLoss.hpp"

void IRegressionLoss::computeGradientsHessians(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients,
    std::vector<double>& hessians) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    hessians.resize(n);
    
    // 默认实现：逐个计算
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = gradient(y_true[i], y_pred[i]);
        hessians[i] = hessian(y_true[i], y_pred[i]);
    }
}