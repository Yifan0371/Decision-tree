#pragma once

#include "IRegressionLoss.hpp"
#include <cmath>


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
            return 0.0;  
        }
    }
    
    double hessian(double , double ) const override {
        return 0.0;  
    }
    
    std::string name() const override { return "absolute"; }
    bool supportsSecondOrder() const override { return false; }
};
