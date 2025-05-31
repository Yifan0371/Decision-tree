
// =============================================================================
// include/boosting/loss/SquaredLoss.hpp
// =============================================================================
#ifndef BOOSTING_LOSS_SQUAREDLOSS_HPP
#define BOOSTING_LOSS_SQUAREDLOSS_HPP

#include "IRegressionLoss.hpp"

/** 平方损失函数：L(y, f) = 0.5 * (y - f)^2 - 最常用的回归损失 */
class SquaredLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override {
        double diff = y_true - y_pred;
        return 0.5 * diff * diff;
    }
    
    double gradient(double y_true, double y_pred) const override {
        return y_true - y_pred;  // 负梯度：-(-(y-f)) = y-f，即残差
    }
    
    double hessian(double /* y_true */, double /* y_pred */) const override {
        return 1.0;  // 二阶导数为常数1
    }
    
    std::string name() const override { return "squared"; }
    bool supportsSecondOrder() const override { return true; }
    
    // 优化的批量计算
    void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const override {
        
        size_t n = y_true.size();
        gradients.resize(n);
        hessians.assign(n, 1.0);  // Hessian为常数，直接赋值
        
        // 向量化残差计算
        for (size_t i = 0; i < n; ++i) {
            gradients[i] = y_true[i] - y_pred[i];
        }
    }
};

#endif // BOOSTING_LOSS_SQUAREDLOSS_HPP