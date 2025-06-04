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
    
    bool isTreeDropped(int treeIndex, const std::vector<int>& droppedIndices) const;
};
