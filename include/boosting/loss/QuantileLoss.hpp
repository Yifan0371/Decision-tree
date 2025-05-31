
// =============================================================================
// include/boosting/loss/QuantileLoss.hpp
// =============================================================================
#ifndef BOOSTING_LOSS_QUANTILELOSS_HPP
#define BOOSTING_LOSS_QUANTILELOSS_HPP

#include "IRegressionLoss.hpp"
#include <cmath>

/** 分位数损失函数：用于分位数回归 */
class QuantileLoss : public IRegressionLoss {
public:
    explicit QuantileLoss(double quantile = 0.5) : quantile_(quantile) {
        // 确保分位数在有效范围内
        if (quantile <= 0.0 || quantile >= 1.0) {
            quantile_ = 0.5;
        }
    }
    
    double loss(double y_true, double y_pred) const override {
        double diff = y_true - y_pred;
        if (diff >= 0) {
            return quantile_ * diff;
        } else {
            return (quantile_ - 1.0) * diff;
        }
    }
    
    double gradient(double y_true, double y_pred) const override {
        double diff = y_true - y_pred;
        if (diff >= 0) {
            return quantile_;
        } else {
            return quantile_ - 1.0;
        }
    }
    
    double hessian(double /* y_true */, double /* y_pred */) const override {
        return 0.0;  // 分位数损失的二阶导数为0
    }
    
    std::string name() const override { 
        return "quantile_" + std::to_string(quantile_); 
    }
    bool supportsSecondOrder() const override { return false; }
    
    double getQuantile() const { return quantile_; }

private:
    double quantile_;
};

#endif // BOOSTING_LOSS_QUANTILELOSS_HPP