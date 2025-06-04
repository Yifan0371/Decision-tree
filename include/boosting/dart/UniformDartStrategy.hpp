#pragma once

#include "IDartStrategy.hpp"
#include <unordered_set>

enum class DartWeightStrategy {
    NONE,           
    MILD,           
    ORIGINAL,       
    EXPERIMENTAL    
};

class UniformDartStrategy : public IDartStrategy {
public:
    explicit UniformDartStrategy(bool normalizeWeights = true, 
                                bool skipDropForPrediction = false,
                                DartWeightStrategy weightStrategy = DartWeightStrategy::MILD)
        : normalizeWeights_(normalizeWeights), 
          skipDropForPrediction_(skipDropForPrediction),
          weightStrategy_(weightStrategy) {}
    
    // **原有接口方法**
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
    
    // **配置方法**
    void setNormalizeWeights(bool normalize) { normalizeWeights_ = normalize; }
    void setSkipDropForPrediction(bool skip) { skipDropForPrediction_ = skip; }

    // **新增：优化的预测方法**
    double computeFullPredictionOptimized(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    double computeDropoutPredictionOptimized(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    // **新增：智能dropout策略方法**
    double computeDropoutByExclusion(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    double computeDropoutByInclusion(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    // **新增：辅助方法**
    double computeSingleTreeContribution(
        const RegressionBoostingModel::RegressionTree& tree,
        const double* sample,
        int rowLength) const;
    
    // **新增：批量处理方法**
    void computeDropoutPredictionsBatch(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const std::vector<double>& X,
        int rowLength,
        double baseScore,
        std::vector<double>& predictions) const;
    
    // **新增：自适应dropout选择**
    std::vector<int> selectDroppedTreesAdaptive(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        double dropRate,
        std::mt19937& gen) const;

private:
    bool normalizeWeights_;
    bool skipDropForPrediction_;
    DartWeightStrategy weightStrategy_;
    
    void updateTreeWeightsStrategy(std::vector<RegressionBoostingModel::RegressionTree>& trees,
                                  const std::vector<int>& droppedIndices,
                                  int newTreeIndex,
                                  double learningRate) const;
    
    bool isTreeDropped(int treeIndex, const std::vector<int>& droppedIndices) const;
};