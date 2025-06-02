// =============================================================================
// src/boosting/dart/UniformDartStrategy.cpp
// =============================================================================
#include "boosting/dart/UniformDartStrategy.hpp"
#include <algorithm>
#include <random>
#include <ostream>
#include <iostream>
#include <cmath>

std::vector<int> UniformDartStrategy::selectDroppedTrees(
    int totalTrees, double dropRate, std::mt19937& gen) const {
    
    if (totalTrees <= 0 || dropRate <= 0.0 || dropRate >= 1.0) {
        return {};
    }
    
    std::vector<int> droppedTrees;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // 均匀随机丢弃
    for (int i = 0; i < totalTrees; ++i) {
        if (dist(gen) < dropRate) {
            droppedTrees.push_back(i);
        }
    }
    
    return droppedTrees;
}

double UniformDartStrategy::computeDropoutPrediction(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int /* rowLength */,  // 使用注释避免警告
    double baseScore) const {
    
    double prediction = baseScore;
    
    // 如果预测时跳过dropout，使用所有树
    if (skipDropForPrediction_) {
        for (const auto& regTree : trees) {
            const Node* cur = regTree.tree.get();
            while (cur && !cur->isLeaf) {
                double value = sample[cur->getFeatureIndex()];
                cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
            }
            if (cur) {
                prediction += regTree.learningRate * regTree.weight * cur->getPrediction();
            }
        }
        return prediction;
    }
    
    // 训练时：排除丢弃的树
    for (size_t i = 0; i < trees.size(); ++i) {
        if (isTreeDropped(static_cast<int>(i), droppedIndices)) {
            continue; // 跳过被丢弃的树
        }
        
        const auto& regTree = trees[i];
        const Node* cur = regTree.tree.get();
        while (cur && !cur->isLeaf) {
            double value = sample[cur->getFeatureIndex()];
            cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        if (cur) {
            prediction += regTree.learningRate * regTree.weight * cur->getPrediction();
        }
    }
    
    return prediction;
}
void UniformDartStrategy::updateTreeWeights(
    std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    int newTreeIndex,
    double learningRate) const {
    
    if (!normalizeWeights_ || trees.empty()) {
        return;
    }
    
    double k = static_cast<double>(droppedIndices.size());
    if (k == 0.0) return;
    
    switch (weightStrategy_) {
        case DartWeightStrategy::NONE:
            // 不做任何权重调整
            break;
            
        case DartWeightStrategy::MILD: {
            // 温和策略：轻微增加新树权重
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                double adjustmentFactor = 1.0 + 0.05 * k; // 每丢弃一棵树，权重增加5%
                trees[newTreeIndex].weight = learningRate * std::min(adjustmentFactor, 1.2);
            }
            break;
        }
        
        case DartWeightStrategy::ORIGINAL: {
            // 原始DART方法：激进调整
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                trees[newTreeIndex].weight = learningRate * (k + 1.0);
            }
            break;
        }
        
        case DartWeightStrategy::EXPERIMENTAL: {
            // 实验性方法：基于总损失减少的权重调整
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                double totalTrees = static_cast<double>(trees.size());
                double dropRatio = k / totalTrees;
                double adaptiveFactor = 1.0 + dropRatio * 0.5; // 自适应调整
                trees[newTreeIndex].weight = learningRate * adaptiveFactor;
            }
            break;
        }
    }
}

bool UniformDartStrategy::isTreeDropped(int treeIndex, 
                                       const std::vector<int>& droppedIndices) const {
    return std::find(droppedIndices.begin(), droppedIndices.end(), treeIndex) 
           != droppedIndices.end();
}