
// =============================================================================
// include/boosting/loss/AbsoluteLoss.hpp
// =============================================================================
#ifndef BOOSTING_LOSS_ABSOLUTELOSS_HPP
#define BOOSTING_LOSS_ABSOLUTELOSS_HPP

#include "IRegressionLoss.hpp"
#include <cmath>

/** 绝对值损失函数：L(y, f) = |y - f| - 对离群点鲁棒，等同于中位数回归 */
class AbsoluteLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override {
        return std::abs(y_true - y_pred);
    }
    
    double gradient(double y_true, double y_pred) const override {
        double diff = y_true - y_pred;
        if (diff > 0) {
            return 1.0;
        } else if (diff < 0) {
            return -1.0;
        } else {
            return 0.0;  // 在y_true == y_pred处，梯度为0
        }
    }
    
    double hessian(double /* y_true */, double /* y_pred */) const override {
        return 0.0;  // 绝对值函数在可导点的二阶导数为0
    }
    
    std::string name() const override { return "absolute"; }
    bool supportsSecondOrder() const override { return false; }
};

#endif // BOOSTING_LOSS_ABSOLUTELOSS_HPP