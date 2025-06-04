#pragma once

#include "boosting/loss/IRegressionLoss.hpp"
#include <memory>
#include <string>
#include <vector>   


class XGBoostLossFactory {
public:
    static std::unique_ptr<IRegressionLoss> create(const std::string& objective);
};


class XGBoostSquaredLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override;
    double gradient(double y_true, double y_pred) const override;
    double hessian(double y_true, double y_pred) const override;
    std::string name() const override { return "xgb:squarederror"; }
    bool supportsSecondOrder() const override { return true; }
    
    
    void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const override;
};


class XGBoostLogisticLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override;
    double gradient(double y_true, double y_pred) const override;
    double hessian(double y_true, double y_pred) const override;
    std::string name() const override { return "xgb:logistic"; }
    bool supportsSecondOrder() const override { return true; }
};
