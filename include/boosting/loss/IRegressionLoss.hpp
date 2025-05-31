
// =============================================================================
// include/boosting/loss/IRegressionLoss.hpp
// =============================================================================
#ifndef BOOSTING_LOSS_IREGRESSIONLOSS_HPP
#define BOOSTING_LOSS_IREGRESSIONLOSS_HPP

#include <vector>
#include <string>

/**
 * 回归损失函数接口：专门为回归任务设计
 * 支持一阶和二阶梯度计算，为XGBoost等高级算法预留接口
 */
class IRegressionLoss {
public:
    virtual ~IRegressionLoss() = default;
    
    /** 计算单个样本的损失值 */
    virtual double loss(double y_true, double y_pred) const = 0;
    
    /** 计算一阶梯度（负梯度作为伪残差） */
    virtual double gradient(double y_true, double y_pred) const = 0;
    
    /** 计算二阶梯度（Hessian对角元素，为XGBoost预留） */
    virtual double hessian(double y_true, double y_pred) const = 0;
    
    /** 
     * 批量计算梯度和Hessian（内存优化版本）
     * 这是性能关键路径，默认实现可被子类优化
     */
    virtual void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const;
    
    /** 获取损失函数名称 */
    virtual std::string name() const = 0;
    
    /** 是否支持二阶优化（XGBoost需要） */
    virtual bool supportsSecondOrder() const { return false; }
};

#endif // BOOSTING_LOSS_IREGRESSIONLOSS_HPP
