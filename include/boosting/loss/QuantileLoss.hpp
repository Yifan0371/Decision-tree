#pragma once

#include "IRegressionLoss.hpp"
#include <cmath>


class QuantileLoss : public IRegressionLoss {
public:
    explicit QuantileLoss(double quantile = 0.5) : quantile_(quantile) {
        
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
    
    double hessian(double , double ) const override {
        return 0.0;  
    }
    
    std::string name() const override { 
        return "quantile_" + std::to_string(quantile_); 
    }
    bool supportsSecondOrder() const override { return false; }
    
    double getQuantile() const { return quantile_; }

private:
    double quantile_;
};
