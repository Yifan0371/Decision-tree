// =============================================================================
// src/boosting/dart/UniformDartStrategy.cpp - 深度并行优化版本
// =============================================================================
#include "boosting/dart/UniformDartStrategy.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <execution>  // C++17并行算法
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<int> UniformDartStrategy::selectDroppedTrees(
    int totalTrees, double dropRate, std::mt19937& gen) const {
    
    if (totalTrees <= 0 || dropRate <= 0.0 || dropRate >= 1.0) {
        return {};
    }
    
    std::vector<int> droppedTrees;
    const int expectedDrops = static_cast<int>(std::ceil(totalTrees * dropRate));
    droppedTrees.reserve(expectedDrops + 5);  // 预分配空间
    
    // **优化1: 批量随机数生成**
    std::vector<double> randomValues(totalTrees);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // 批量生成随机数（更高效）
    for (int i = 0; i < totalTrees; ++i) {
        randomValues[i] = dist(gen);
    }
    
    // **优化2: 向量化的选择过程**
    for (int i = 0; i < totalTrees; ++i) {
        if (randomValues[i] < dropRate) {
            droppedTrees.push_back(i);
        }
    }
    
    // **优化3: 确保至少丢弃一些树（如果期望值>=1）**
    if (droppedTrees.empty() && expectedDrops >= 1 && totalTrees > 0) {
        // 强制丢弃一棵随机树
        std::uniform_int_distribution<int> treeDist(0, totalTrees - 1);
        droppedTrees.push_back(treeDist(gen));
    }
    
    return droppedTrees;
}

double UniformDartStrategy::computeDropoutPrediction(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    // **预测时跳过dropout（用于最终预测）**
    if (skipDropForPrediction_) {
        return computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    }
    
    // **训练时排除丢弃的树**
    return computeDropoutPredictionOptimized(trees, droppedIndices, sample, rowLength, baseScore);
}

// **优化1: 高效的完整预测（无dropout）**
double UniformDartStrategy::computeFullPredictionOptimized(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    double prediction = baseScore;
    
    // **展开少量循环，减少分支预测失误**
    const size_t numTrees = trees.size();
    size_t i = 0;
    
    // **批处理前几棵树（展开循环）**
    for (; i + 3 < numTrees; i += 4) {
        prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+1], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+2], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+3], sample, rowLength);
    }
    
    // **处理剩余的树**
    for (; i < numTrees; ++i) {
        prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
    }
    
    return prediction;
}

// **优化2: 高效的dropout预测**
double UniformDartStrategy::computeDropoutPredictionOptimized(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    if (droppedIndices.empty()) {
        return computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    }
    
    double prediction = baseScore;
    
    // **优化: 如果丢弃的树很少，使用排除法**
    if (droppedIndices.size() <= 5) {
        return computeDropoutByExclusion(trees, droppedIndices, sample, rowLength, baseScore);
    }
    
    // **优化: 如果丢弃的树很多，使用包含法**
    return computeDropoutByInclusion(trees, droppedIndices, sample, rowLength, baseScore);
}

// **丢弃树较少时：排除法（计算全部然后减去丢弃的）**
double UniformDartStrategy::computeDropoutByExclusion(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    // 先计算完整预测
    double fullPrediction = computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    
    // 减去被丢弃树的贡献
    for (int idx : droppedIndices) {
        if (idx >= 0 && idx < static_cast<int>(trees.size())) {
            fullPrediction -= computeSingleTreeContribution(trees[idx], sample, rowLength);
        }
    }
    
    return fullPrediction;
}

// **丢弃树较多时：包含法（只计算保留的树）**
double UniformDartStrategy::computeDropoutByInclusion(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    // **创建快速查找表**
    static thread_local std::vector<bool> droppedMask;
    const size_t numTrees = trees.size();
    
    if (droppedMask.size() != numTrees) {
        droppedMask.resize(numTrees);
    }
    std::fill(droppedMask.begin(), droppedMask.end(), false);
    
    // **标记丢弃的树**
    for (int idx : droppedIndices) {
        if (idx >= 0 && idx < static_cast<int>(numTrees)) {
            droppedMask[idx] = true;
        }
    }
    
    // **只累加未丢弃的树**
    double prediction = baseScore;
    for (size_t i = 0; i < numTrees; ++i) {
        if (!droppedMask[i]) {
            prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
        }
    }
    
    return prediction;
}

// **内联的单树贡献计算**
inline double UniformDartStrategy::computeSingleTreeContribution(
    const RegressionBoostingModel::RegressionTree& tree,
    const double* sample,
    int rowLength) const {
    
    // **快速树遍历**
    const Node* cur = tree.tree.get();
    while (cur && !cur->isLeaf) {
        const int featIdx = cur->getFeatureIndex();
        const double threshold = cur->getThreshold();
        cur = (sample[featIdx] <= threshold) ? cur->getLeft() : cur->getRight();
    }
    
    const double treePred = cur ? cur->getPrediction() : 0.0;
    return tree.learningRate * tree.weight * treePred;
}

void UniformDartStrategy::updateTreeWeights(
    std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    int newTreeIndex,
    double learningRate) const {
    
    if (!normalizeWeights_ || trees.empty()) {
        return;
    }
    
    const double k = static_cast<double>(droppedIndices.size());
    if (k == 0.0) return;
    
    // **优化: 批量权重更新**
    switch (weightStrategy_) {
        case DartWeightStrategy::NONE:
            // 不做任何权重调整
            break;
            
        case DartWeightStrategy::MILD: {
            // 温和策略：轻微增加新树权重
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                const double adjustmentFactor = 1.0 + 0.05 * k;
                trees[newTreeIndex].weight = learningRate * std::min(adjustmentFactor, 1.2);
            }
            break;
        }
        
        case DartWeightStrategy::ORIGINAL: {
            // 原始DART方法：激进调整
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                trees[newTreeIndex].weight = learningRate * (k + 1.0);
            }
            
            // **优化: 并行更新丢弃树的权重**
            if (droppedIndices.size() > 10) {
                #pragma omp parallel for schedule(static) if(droppedIndices.size() > 50)
                for (size_t i = 0; i < droppedIndices.size(); ++i) {
                    const int idx = droppedIndices[i];
                    if (idx >= 0 && idx < static_cast<int>(trees.size())) {
                        trees[idx].weight *= (k + 1.0) / k;  // 重新平衡权重
                    }
                }
            } else {
                // 小批量使用串行更新
                for (int idx : droppedIndices) {
                    if (idx >= 0 && idx < static_cast<int>(trees.size())) {
                        trees[idx].weight *= (k + 1.0) / k;
                    }
                }
            }
            break;
        }
        
        case DartWeightStrategy::EXPERIMENTAL: {
            // 实验性方法：自适应权重调整
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                const double totalTrees = static_cast<double>(trees.size());
                const double dropRatio = k / totalTrees;
                const double adaptiveFactor = 1.0 + dropRatio * 0.5;
                trees[newTreeIndex].weight = learningRate * adaptiveFactor;
                
                // **实验性: 动态调整学习率衰减**
                const double decayFactor = std::max(0.95, 1.0 - dropRatio * 0.1);
                trees[newTreeIndex].learningRate *= decayFactor;
            }
            break;
        }
    }
}

// **优化的树查找**
bool UniformDartStrategy::isTreeDropped(int treeIndex, 
                                       const std::vector<int>& droppedIndices) const {
    
    // **优化: 对于小的丢弃列表，使用线性搜索**
    if (droppedIndices.size() <= 8) {
        for (int idx : droppedIndices) {
            if (idx == treeIndex) return true;
        }
        return false;
    }
    
    // **优化: 对于大的丢弃列表，使用二分搜索**
    // 注意：这要求droppedIndices是有序的，如果不是，需要先排序
    return std::binary_search(droppedIndices.begin(), droppedIndices.end(), treeIndex);
}

// **新增: 批量dropout预测（用于训练时批量处理）**
void UniformDartStrategy::computeDropoutPredictionsBatch(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const std::vector<double>& X,
    int rowLength,
    double baseScore,
    std::vector<double>& predictions) const {
    
    const size_t n = predictions.size();
    
    if (droppedIndices.empty()) {
        // **无dropout时的批量预测**
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            predictions[i] = computeFullPredictionOptimized(
                trees, &X[i * rowLength], rowLength, baseScore);
        }
    } else {
        // **有dropout时的批量预测**
        #pragma omp parallel for schedule(static, 256) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            predictions[i] = computeDropoutPredictionOptimized(
                trees, droppedIndices, &X[i * rowLength], rowLength, baseScore);
        }
    }
}

// **新增: 智能dropout策略（根据树的重要性选择丢弃）**
std::vector<int> UniformDartStrategy::selectDroppedTreesAdaptive(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    double dropRate,
    std::mt19937& gen) const {
    
    const int totalTrees = static_cast<int>(trees.size());
    if (totalTrees <= 0 || dropRate <= 0.0) {
        return {};
    }
    
    // **计算每棵树的重要性权重**
    std::vector<double> treeWeights(totalTrees);
    for (int i = 0; i < totalTrees; ++i) {
        treeWeights[i] = std::abs(trees[i].weight * trees[i].learningRate);
    }
    
    // **创建加权分布**
    std::discrete_distribution<int> dist(treeWeights.begin(), treeWeights.end());
    
    const int numToDrop = static_cast<int>(std::ceil(totalTrees * dropRate));
    std::vector<int> droppedTrees;
    std::vector<bool> alreadyDropped(totalTrees, false);
    
    // **按权重随机选择要丢弃的树**
    for (int i = 0; i < numToDrop && droppedTrees.size() < static_cast<size_t>(totalTrees); ++i) {
        int candidate = dist(gen);
        if (!alreadyDropped[candidate]) {
            droppedTrees.push_back(candidate);
            alreadyDropped[candidate] = true;
        }
    }
    
    return droppedTrees;
}