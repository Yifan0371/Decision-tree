// =============================================================================
// src/xgboost/loss/XGBoostLossFactory.cpp
// =============================================================================
#include "xgboost/loss/XGBoostLossFactory.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "boosting/loss/HuberLoss.hpp"
#include <stdexcept>
#include <cmath>

std::unique_ptr<IRegressionLoss> XGBoostLossFactory::create(const std::string& objective) {
    if (objective == "reg:squarederror" || objective == "reg:linear") {
        return std::make_unique<XGBoostSquaredLoss>();
    }
    else if (objective == "reg:logistic" || objective == "binary:logistic") {
        return std::make_unique<XGBoostLogisticLoss>();
    }
    else if (objective == "reg:squaredlogerror") {
        return std::make_unique<XGBoostSquaredLoss>();  // 简化实现
    }
    else {
        throw std::invalid_argument("Unsupported objective: " + objective);
    }
}

// XGBoostSquaredLoss实现
double XGBoostSquaredLoss::loss(double y_true, double y_pred) const {
    double diff = y_true - y_pred;
    return 0.5 * diff * diff;
}

double XGBoostSquaredLoss::gradient(double y_true, double y_pred) const {
    return y_pred - y_true;  // 一阶导数
}

double XGBoostSquaredLoss::hessian(double /* y_true */, double /* y_pred */) const {
    return 1.0;  // 二阶导数为常数1
}

void XGBoostSquaredLoss::computeGradientsHessians(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients,
    std::vector<double>& hessians) const {

    size_t n = y_true.size();
    gradients.resize(n);
    hessians.assign(n, 1.0);

    for (size_t i = 0; i < n; ++i) {
        gradients[i] = y_pred[i] - y_true[i];   // ← 方向一致
    }
}

// XGBoostLogisticLoss实现
double XGBoostLogisticLoss::loss(double y_true, double y_pred) const {
    // 防止数值溢出
    double z = std::max(-250.0, std::min(250.0, y_pred));
    return y_true * std::log(1 + std::exp(-z)) + (1 - y_true) * std::log(1 + std::exp(z));
}

double XGBoostLogisticLoss::gradient(double y_true, double y_pred) const {
    double z = std::max(-250.0, std::min(250.0, y_pred));
    double prob = 1.0 / (1.0 + std::exp(-z));
    return prob - y_true;
}

double XGBoostLogisticLoss::hessian(double /* y_true */, double y_pred) const {
    double z = std::max(-250.0, std::min(250.0, y_pred));
    double prob = 1.0 / (1.0 + std::exp(-z));
    return prob * (1.0 - prob);
}