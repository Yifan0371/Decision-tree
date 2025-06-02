// =============================================================================
// include/boosting/dart/IDartStrategy.hpp
// =============================================================================
#ifndef BOOSTING_DART_IDARTSTRATEGY_HPP
#define BOOSTING_DART_IDARTSTRATEGY_HPP

#include "boosting/model/RegressionBoostingModel.hpp"
#include <vector>
#include <memory>
#include <random>

/**
 * DART策略接口：定义dropout行为
 * DART (Dropouts meet Multiple Additive Regression Trees)
 */
class IDartStrategy {
public:
    virtual ~IDartStrategy() = default;
    
    /** 
     * 选择要在当前轮次丢弃的树索引
     * @param totalTrees 当前模型中的树总数
     * @param dropRate 丢弃率 [0.0, 1.0]
     * @param gen 随机数生成器
     * @return 被丢弃的树索引列表
     */
    virtual std::vector<int> selectDroppedTrees(
        int totalTrees, 
        double dropRate,
        std::mt19937& gen) const = 0;
    
    /** 
     * 计算排除丢弃树后的预测值
     * @param trees 所有树的信息
     * @param droppedIndices 被丢弃的树索引
     * @param sample 输入样本
     * @param rowLength 特征数量
     * @param baseScore 基准分数
     * @return 预测值
     */
    virtual double computeDropoutPrediction(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const = 0;
    
    /** 
     * 训练后更新树权重（DART归一化）
     * @param trees 所有树的引用
     * @param droppedIndices 本轮丢弃的树索引
     * @param newTreeIndex 新添加的树索引
     * @param learningRate 学习率
     */
    virtual void updateTreeWeights(
        std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        int newTreeIndex,
        double learningRate) const = 0;
    
    /** 获取策略名称 */
    virtual std::string name() const = 0;
};

#endif // BOOSTING_DART_IDARTSTRATEGY_HPP