// =============================================================================
// include/boosting/dart/UniformDartStrategy.hpp
// =============================================================================
#ifndef BOOSTING_DART_UNIFORMDARTSTRATEGY_HPP
#define BOOSTING_DART_UNIFORMDARTSTRATEGY_HPP

#include "IDartStrategy.hpp"
#include <unordered_set>

enum class DartWeightStrategy {
    NONE,           // 不做权重调整
    MILD,           // 温和调整
    ORIGINAL,       // 原始DART论文方法
    EXPERIMENTAL    // 实验性方法
};
/**
 * 均匀DART策略：每棵树以相同概率被丢弃
 * 这是最常用的DART实现方式
 */

class UniformDartStrategy : public IDartStrategy {
public:
    explicit UniformDartStrategy(bool normalizeWeights = true, 
                                bool skipDropForPrediction = false,
                                DartWeightStrategy weightStrategy = DartWeightStrategy::MILD)
        : normalizeWeights_(normalizeWeights), 
          skipDropForPrediction_(skipDropForPrediction),
          weightStrategy_(weightStrategy) {}
    
    std::vector<int> selectDroppedTrees(int totalTrees, double dropRate, 
                                       std::mt19937& gen) const override;
    
    double computeDropoutPrediction(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const override;
    
    void updateTreeWeights(
        std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        int newTreeIndex,
        double learningRate) const override;
    
    std::string name() const override { return "uniform_dart"; }
    
    // 配置选项
    void setNormalizeWeights(bool normalize) { normalizeWeights_ = normalize; }
    void setSkipDropForPrediction(bool skip) { skipDropForPrediction_ = skip; }

private:
    bool normalizeWeights_;
    bool skipDropForPrediction_;
    DartWeightStrategy weightStrategy_;
    
    void updateTreeWeightsStrategy(std::vector<RegressionBoostingModel::RegressionTree>& trees,
                                  const std::vector<int>& droppedIndices,
                                  int newTreeIndex,
                                  double learningRate) const;
    /** 检查树索引是否在丢弃列表中 */
    bool isTreeDropped(int treeIndex, const std::vector<int>& droppedIndices) const;
};

#endif // BOOSTING_DART_UNIFORMDARTSTRATEGY_HPP