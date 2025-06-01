
// =============================================================================
// include/xgboost/loss/XGBoostLossFactory.hpp  
// =============================================================================
#ifndef XGBOOST_LOSS_XGBOOSTLOSSFACTORY_HPP
#define XGBOOST_LOSS_XGBOOSTLOSSFACTORY_HPP

#include "boosting/loss/IRegressionLoss.hpp"
#include <memory>
#include <string>
#include <vector>   

/** XGBoost损失函数工厂 - 确保所有损失函数都支持二阶导数 */
class XGBoostLossFactory {
public:
    static std::unique_ptr<IRegressionLoss> create(const std::string& objective);
};

/** XGBoost增强版平方损失 - 针对XGBoost优化 */
class XGBoostSquaredLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override;
    double gradient(double y_true, double y_pred) const override;
    double hessian(double y_true, double y_pred) const override;
    std::string name() const override { return "xgb:squarederror"; }
    bool supportsSecondOrder() const override { return true; }
    
    // XGBoost专用：批量计算（向量化优化）
    void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const override;
};

/** XGBoost Logistic损失（为二分类预留） */
class XGBoostLogisticLoss : public IRegressionLoss {
public:
    double loss(double y_true, double y_pred) const override;
    double gradient(double y_true, double y_pred) const override;
    double hessian(double y_true, double y_pred) const override;
    std::string name() const override { return "xgb:logistic"; }
    bool supportsSecondOrder() const override { return true; }
};

#endif // XGBOOST_LOSS_XGBOOSTLOSSFACTORY_HPP
