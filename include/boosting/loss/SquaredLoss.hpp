#pragma once

#include "IRegressionLoss.hpp"


class SquaredLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override {
        double diff = y_true - y_pred;
        return 0.5 * diff * diff;
    }
    
    double gradient(double y_true, double y_pred) const override {
        return y_true - y_pred;  
    }
    
    double hessian(double , double ) const override {
        return 1.0;  
    }
    
    std::string name() const override { return "squared"; }
    bool supportsSecondOrder() const override { return true; }
    
    
    void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const override {
        
        size_t n = y_true.size();
        gradients.resize(n);
        hessians.assign(n, 1.0);  
        
        
        for (size_t i = 0; i < n; ++i) {
            gradients[i] = y_true[i] - y_pred[i];
        }
    }
};
