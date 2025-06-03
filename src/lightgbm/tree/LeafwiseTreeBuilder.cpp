// =============================================================================
// src/lightgbm/tree/LeafwiseTreeBuilder.cpp - OpenMP深度并行优化版本
// =============================================================================
#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::unique_ptr<Node> LeafwiseTreeBuilder::buildTree(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& /* labels */,
    const std::vector<double>& targets, // 残差
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights,
    const std::vector<FeatureBundle>& /* bundles */) {
    
    while (!leafQueue_.empty()) leafQueue_.pop();
    
    auto root = std::make_unique<Node>();
    root->samples = sampleIndices.size();
    
    // **并行优化1: GOSS权重兼容的根节点预测计算**
    double weightedSum = 0.0, totalWeight = 0.0;
    size_t n = sampleIndices.size();
    
    if (n > 1000) {
        // 并行版本
        #pragma omp parallel for reduction(+:weightedSum,totalWeight) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double weight = sampleWeights[i];
            weightedSum += targets[idx] * weight;
            totalWeight += weight;
        }
    } else {
        // 串行版本（小数据集）
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double weight = sampleWeights[i];
            weightedSum += targets[idx] * weight;
            totalWeight += weight;
        }
    }
    
    double rootPrediction = (totalWeight > 0) ? weightedSum / totalWeight : 0.0;
    
    // 尝试分裂根节点
    LeafInfo rootInfo;
    rootInfo.node = root.get();
    rootInfo.sampleIndices = sampleIndices;
    
    if (findBestSplitParallel(data, rowLength, targets, rootInfo.sampleIndices, 
                             sampleWeights, rootInfo)) {
        leafQueue_.push(rootInfo);
    } else {
        root->makeLeaf(rootPrediction);
        return root;
    }
    
    // **并行优化2: Leaf-wise生长的智能并行**
    int currentLeaves = 1;
    while (!leafQueue_.empty() && currentLeaves < config_.numLeaves) {
        LeafInfo bestLeaf = leafQueue_.top();
        leafQueue_.pop();
        
        // 分裂条件检查
        if (bestLeaf.splitGain <= config_.minSplitGain ||
            bestLeaf.sampleIndices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
            
            // **并行优化3: 叶子预测值计算的并行**
            double leafPred = computeLeafPredictionParallel(bestLeaf.sampleIndices, 
                                                           targets, sampleWeights);
            bestLeaf.node->makeLeaf(leafPred);
            continue;
        }
        
        // 执行分裂
        splitLeafParallel(bestLeaf, data, rowLength, targets, sampleWeights);
        currentLeaves += 1;
    }
    
    // **并行优化4: 剩余节点处理的并行**
    processRemainingLeavesParallel(targets, sampleWeights);
    
    return root;
}

// **新增方法：并行分裂查找**
bool LeafwiseTreeBuilder::findBestSplitParallel(const std::vector<double>& data,
                                               int rowLength,
                                               const std::vector<double>& targets,
                                               const std::vector<int>& indices,
                                               const std::vector<double>& weights,
                                               LeafInfo& leafInfo) {
    
    if (indices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
        return false;
    }
    
    // **并行优化5: 当前度量计算的优化**
    // 这里简化处理，使用原始targets进行分裂评估
    double currentMetric = criterion_->nodeMetric(targets, indices);
    
    auto [bestFeature, bestThreshold, bestGain] = 
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    
    leafInfo.bestFeature = bestFeature;
    leafInfo.bestThreshold = bestThreshold;
    leafInfo.splitGain = bestGain;
    
    return bestFeature >= 0 && bestGain > 0;
}

// **新增方法：并行叶子分裂**
void LeafwiseTreeBuilder::splitLeafParallel(LeafInfo& leafInfo,
                                           const std::vector<double>& data,
                                           int rowLength,
                                           const std::vector<double>& targets,
                                           const std::vector<double>& sampleWeights) {
    
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();
    
    // **并行优化6: 样本分割的并行**
    leftIndices_.clear();
    rightIndices_.clear();
    std::vector<double> leftWeights, rightWeights;
    
    size_t n = leafInfo.sampleIndices.size();
    leftIndices_.reserve(n / 2);
    rightIndices_.reserve(n / 2);
    leftWeights.reserve(n / 2);
    rightWeights.reserve(n / 2);
    
    // 根据数据大小选择并行策略
    if (n > 1000) {
        // 大数据集：先并行分类，再串行收集
        std::vector<char> leftMask(n);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int idx = leafInfo.sampleIndices[i];
            double value = data[idx * rowLength + leafInfo.bestFeature];
            leftMask[i] = (value <= leafInfo.bestThreshold) ? 1 : 0;
        }
        
        // 串行收集（避免竞争条件）
        for (size_t i = 0; i < n; ++i) {
            int idx = leafInfo.sampleIndices[i];
            double weight = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
            
            if (leftMask[i]) {
                leftIndices_.push_back(idx);
                leftWeights.push_back(weight);
            } else {
                rightIndices_.push_back(idx);
                rightWeights.push_back(weight);
            }
        }
    } else {
        // 小数据集：串行处理
        for (size_t i = 0; i < n; ++i) {
            int idx = leafInfo.sampleIndices[i];
            double value = data[idx * rowLength + leafInfo.bestFeature];
            double weight = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
            
            if (value <= leafInfo.bestThreshold) {
                leftIndices_.push_back(idx);
                leftWeights.push_back(weight);
            } else {
                rightIndices_.push_back(idx);
                rightWeights.push_back(weight);
            }
        }
    }
    
    // **并行优化7: 左右子节点的并行处理**
    // 使用并行sections同时处理左右子节点
    #pragma omp parallel sections if(leftIndices_.size() > 500 || rightIndices_.size() > 500)
    {
        #pragma omp section
        {
            // 处理左子节点
            if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
                LeafInfo leftInfo;
                leftInfo.node = leafInfo.node->leftChild.get();
                leftInfo.sampleIndices = leftIndices_;
                leftInfo.node->samples = leftIndices_.size();
                
                if (findBestSplitParallel(data, rowLength, targets, 
                                         leftInfo.sampleIndices, leftWeights, leftInfo)) {
                    #pragma omp critical(leaf_queue)
                    {
                        leafQueue_.push(leftInfo);
                    }
                } else {
                    double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights);
                    leftInfo.node->makeLeaf(leftPred);
                }
            } else {
                double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights);
                leafInfo.node->leftChild->makeLeaf(leftPred);
            }
        }
        
        #pragma omp section
        {
            // 处理右子节点
            if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
                LeafInfo rightInfo;
                rightInfo.node = leafInfo.node->rightChild.get();
                rightInfo.sampleIndices = rightIndices_;
                rightInfo.node->samples = rightIndices_.size();
                
                if (findBestSplitParallel(data, rowLength, targets, 
                                         rightInfo.sampleIndices, rightWeights, rightInfo)) {
                    #pragma omp critical(leaf_queue)
                    {
                        leafQueue_.push(rightInfo);
                    }
                } else {
                    double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights);
                    rightInfo.node->makeLeaf(rightPred);
                }
            } else {
                double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights);
                leafInfo.node->rightChild->makeLeaf(rightPred);
            }
        }
    }
}

// **新增方法：并行叶子预测计算**
double LeafwiseTreeBuilder::computeLeafPredictionParallel(
    const std::vector<int>& indices,
    const std::vector<double>& targets,
    const std::vector<double>& weights) const {
    
    if (indices.empty()) return 0.0;
    
    double leafSum = 0.0, leafWeight = 0.0;
    size_t n = indices.size();
    
    if (n > 500) {
        // 并行版本
        #pragma omp parallel for reduction(+:leafSum,leafWeight) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int idx = indices[i];
            double weight = (i < weights.size()) ? weights[i] : 1.0;
            leafSum += targets[idx] * weight;
            leafWeight += weight;
        }
    } else {
        // 串行版本
        for (size_t i = 0; i < n; ++i) {
            int idx = indices[i];
            double weight = (i < weights.size()) ? weights[i] : 1.0;
            leafSum += targets[idx] * weight;
            leafWeight += weight;
        }
    }
    
    return (leafWeight > 0) ? leafSum / leafWeight : 0.0;
}

// **新增方法：并行处理剩余叶子**
void LeafwiseTreeBuilder::processRemainingLeavesParallel(
    const std::vector<double>& targets,
    const std::vector<double>& sampleWeights) {
    
    // 收集所有剩余的叶子节点
    std::vector<LeafInfo> remainingLeaves;
    while (!leafQueue_.empty()) {
        remainingLeaves.push_back(leafQueue_.top());
        leafQueue_.pop();
    }
    
    if (remainingLeaves.empty()) return;
    
    // **并行优化8: 剩余叶子的并行处理**
    #pragma omp parallel for schedule(dynamic) if(remainingLeaves.size() > 4)
    for (size_t i = 0; i < remainingLeaves.size(); ++i) {
        const auto& remaining = remainingLeaves[i];
        double leafPred = computeLeafPredictionParallel(remaining.sampleIndices, 
                                                       targets, sampleWeights);
        remaining.node->makeLeaf(leafPred);
    }
}

// 保留原有方法作为兼容性接口
bool LeafwiseTreeBuilder::findBestSplit(const std::vector<double>& data,
                                       int rowLength,
                                       const std::vector<double>& targets,
                                       const std::vector<int>& indices,
                                       const std::vector<double>& weights,
                                       LeafInfo& leafInfo) {
    return findBestSplitParallel(data, rowLength, targets, indices, weights, leafInfo);
}

void LeafwiseTreeBuilder::splitLeaf(LeafInfo& leafInfo,
                                   const std::vector<double>& data,
                                   int rowLength,
                                   const std::vector<double>& targets,
                                   const std::vector<double>& sampleWeights) {
    splitLeafParallel(leafInfo, data, rowLength, targets, sampleWeights);
}