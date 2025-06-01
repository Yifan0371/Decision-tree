// 1. 修复 include/xgboost/criterion/XGBoostCriterion.hpp
#ifndef XGBOOST_CRITERION_XGBOOSTCRITERION_HPP
#define XGBOOST_CRITERION_XGBOOSTCRITERION_HPP

#include "tree/ISplitCriterion.hpp"
#include <vector>

/** XGBoost分裂准则：基于梯度和Hessian计算结构分数 */
class XGBoostCriterion : public ISplitCriterion {
public:
    explicit XGBoostCriterion(double lambda = 1.0) : lambda_(lambda) {}
    
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override {
        // 对于XGBoost，这个方法不直接使用
        return 0.0;
    }
    
    /** XGBoost专用：计算结构分数 - 修复公式 */
    double computeStructureScore(double G, double H) const {
        // 修复：XGBoost的结构分数应该是正数
        return 0.5 * (G * G) / (H + lambda_);
    }
    
    /** 计算分裂增益 - 修复公式 */
    double computeSplitGain(double Gl, double Hl,
                                double Gr, double Hr,
                                double Gp, double Hp,
                                double gamma) const {
        double gain =
            computeStructureScore(Gl, Hl) +
            computeStructureScore(Gr, Hr) -
            computeStructureScore(Gp, Hp);
        return gain - gamma;
    }

    
    /** 计算最优叶节点权重 */
    double computeLeafWeight(double G, double H) const {
        return -G / (H + lambda_);
    }
    
    double getLambda() const { return lambda_; }

private:
    double lambda_;  // L2正则化参数
};

#endif // XGBOOST_CRITERION_XGBOOSTCRITERION_HPP