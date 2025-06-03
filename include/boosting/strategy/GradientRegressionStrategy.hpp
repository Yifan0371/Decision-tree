// =============================================================================
// include/boosting/strategy/GradientRegressionStrategy.hpp
// =============================================================================
#ifndef BOOSTING_STRATEGY_GRADIENTREGRESSIONSTRATEGY_HPP
#define BOOSTING_STRATEGY_GRADIENTREGRESSIONSTRATEGY_HPP

#include "../loss/IRegressionLoss.hpp"
#include <memory>
#include <vector>
#include <string>

/** 梯度回归Boosting策略：专门为回归任务设计的梯度下降策略 */
class GradientRegressionStrategy {
public:
    explicit GradientRegressionStrategy(
        std::unique_ptr<IRegressionLoss> lossFunc,
        double baseLearningRate = 0.1,
        bool useLineSearch = false)
        : lossFunc_(std::move(lossFunc)),
          baseLearningRate_(baseLearningRate),
          useLineSearch_(useLineSearch) {}

    /** 更新训练目标（计算残差或梯度） */
    void updateTargets(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& targets) const;

    /** 计算当前迭代的学习率 */
    double computeLearningRate(
        int /* iteration */,
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const {
        if (!useLineSearch_) {
            return baseLearningRate_;
        }
        // 线搜索优化学习率
        return computeOptimalLearningRate(y_true, y_pred, tree_pred);
    }

    /** 更新模型预测值 */
    void updatePredictions(
        const std::vector<double>& tree_pred,
        double learning_rate,
        std::vector<double>& y_pred) const;

    std::string name() const { return "gradient_regression"; }

    const IRegressionLoss* getLossFunction() const { return lossFunc_.get(); }

    /** 计算当前预测的总损失 */
    double computeTotalLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const;

private:
    std::unique_ptr<IRegressionLoss> lossFunc_;
    double baseLearningRate_;
    bool useLineSearch_;

    /** 线搜索计算最优学习率 */
    double computeOptimalLearningRate(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const;

    /** 评估给定学习率下的损失 */
    double evaluateLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred,
        double lr) const;
};

#endif // BOOSTING_STRATEGY_GRADIENTREGRESSIONSTRATEGY_HPP
