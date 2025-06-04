#pragma once

#include "IRegressionLoss.hpp"
#include <cmath>


class HuberLoss : public IRegressionLoss {
public:
    explicit HuberLoss(double delta = 1.0) : delta_(delta) {}
    
    double loss(double y_true, double y_pred) const override {
        double r = std::abs(y_true - y_pred);
        if (r <= delta_) {
            return 0.5 * r * r;
        } else {
            return delta_ * (r - 0.5 * delta_);
        }
    }
    
    double gradient(double y_true, double y_pred) const override {
        double r = y_true - y_pred;
        if (std::abs(r) <= delta_) {
            return r;  
        } else {
            return r > 0 ? delta_ : -delta_;  
        }
    }
    
    double hessian(double y_true, double y_pred) const override {
        double r = std::abs(y_true - y_pred);
        return r <= delta_ ? 1.0 : 0.0;
    }
    
    std::string name() const override { return "huber"; }
    bool supportsSecondOrder() const override { return true; }
    
    double getDelta() const { return delta_; }

private:
    double delta_;
};
