#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

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
    
    // **GOSS权重兼容：加权平均计算根节点预测**
    double weightedSum = 0.0, totalWeight = 0.0;
    for (size_t i = 0; i < sampleIndices.size(); ++i) {
        int idx = sampleIndices[i];
        double weight = sampleWeights[i];
        weightedSum += targets[idx] * weight;
        totalWeight += weight;
    }
    double rootPrediction = (totalWeight > 0) ? weightedSum / totalWeight : 0.0;
    
    // 尝试分裂根节点
    LeafInfo rootInfo;
    rootInfo.node = root.get();
    rootInfo.sampleIndices = sampleIndices;
    
    if (findBestSplit(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
        leafQueue_.push(rootInfo);
    } else {
        root->makeLeaf(rootPrediction);
        return root;
    }
    
    // **Leaf-wise生长**
    int currentLeaves = 1;
    while (!leafQueue_.empty() && currentLeaves < config_.numLeaves) {
        LeafInfo bestLeaf = leafQueue_.top();
        leafQueue_.pop();
        
        // 分裂条件检查
        if (bestLeaf.splitGain <= config_.minSplitGain ||
            bestLeaf.sampleIndices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
            
            // 计算加权叶子预测值
            double leafSum = 0.0, leafWeight = 0.0;
            for (size_t i = 0; i < bestLeaf.sampleIndices.size(); ++i) {
                int idx = bestLeaf.sampleIndices[i];
                // **关键：在叶子预测中也要考虑权重**
                double weight = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
                leafSum += targets[idx] * weight;
                leafWeight += weight;
            }
            double leafPred = (leafWeight > 0) ? leafSum / leafWeight : 0.0;
            bestLeaf.node->makeLeaf(leafPred);
            continue;
        }
        
        // 执行分裂
        splitLeaf(bestLeaf, data, rowLength, targets, sampleWeights);
        currentLeaves += 1;
    }
    
    // 处理剩余节点
    while (!leafQueue_.empty()) {
        LeafInfo remaining = leafQueue_.top();
        leafQueue_.pop();
        
        double leafSum = 0.0, leafWeight = 0.0;
        for (size_t i = 0; i < remaining.sampleIndices.size(); ++i) {
            int idx = remaining.sampleIndices[i];
            double weight = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
            leafSum += targets[idx] * weight;
            leafWeight += weight;
        }
        double leafPred = (leafWeight > 0) ? leafSum / leafWeight : 0.0;
        remaining.node->makeLeaf(leafPred);
    }
    
    return root;
}

bool LeafwiseTreeBuilder::findBestSplit(const std::vector<double>& data,
                                       int rowLength,
                                       const std::vector<double>& targets,
                                       const std::vector<int>& indices,
                                       const std::vector<double>& weights,
                                       LeafInfo& leafInfo) {
    
    if (indices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
        return false;
    }
    
    // **简化：使用原始targets进行分裂评估**
    // 这里不完美模拟权重，但避免复杂的权重扩展
    double currentMetric = criterion_->nodeMetric(targets, indices);
    
    auto [bestFeature, bestThreshold, bestGain] = 
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    
    leafInfo.bestFeature = bestFeature;
    leafInfo.bestThreshold = bestThreshold;
    leafInfo.splitGain = bestGain;
    
    return bestFeature >= 0 && bestGain > 0;
}

void LeafwiseTreeBuilder::splitLeaf(LeafInfo& leafInfo,
                                   const std::vector<double>& data,
                                   int rowLength,
                                   const std::vector<double>& targets,
                                   const std::vector<double>& sampleWeights) {
    
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();
    
    // 分割样本
    leftIndices_.clear();
    rightIndices_.clear();
    std::vector<double> leftWeights, rightWeights;
    
    for (size_t i = 0; i < leafInfo.sampleIndices.size(); ++i) {
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
    
    // 创建左子节点
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        
        if (findBestSplit(data, rowLength, targets, leftInfo.sampleIndices, leftWeights, leftInfo)) {
            leafQueue_.push(leftInfo);
        } else {
            // 创建叶子：加权平均
            double leftSum = 0.0, leftWeight = 0.0;
            for (size_t i = 0; i < leftIndices_.size(); ++i) {
                leftSum += targets[leftIndices_[i]] * leftWeights[i];
                leftWeight += leftWeights[i];
            }
            double leftPred = (leftWeight > 0) ? leftSum / leftWeight : 0.0;
            leftInfo.node->makeLeaf(leftPred);
        }
    } else {
        // 样本不足，直接叶子
        double leftSum = 0.0, leftWeight = 0.0;
        for (size_t i = 0; i < leftIndices_.size(); ++i) {
            double weight = (i < leftWeights.size()) ? leftWeights[i] : 1.0;
            leftSum += targets[leftIndices_[i]] * weight;
            leftWeight += weight;
        }
        double leftPred = (leftWeight > 0) ? leftSum / leftWeight : 0.0;
        leafInfo.node->leftChild->makeLeaf(leftPred);
    }
    
    // 创建右子节点（逻辑同左）
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        
        if (findBestSplit(data, rowLength, targets, rightInfo.sampleIndices, rightWeights, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightSum = 0.0, rightWeight = 0.0;
            for (size_t i = 0; i < rightIndices_.size(); ++i) {
                rightSum += targets[rightIndices_[i]] * rightWeights[i];
                rightWeight += rightWeights[i];
            }
            double rightPred = (rightWeight > 0) ? rightSum / rightWeight : 0.0;
            rightInfo.node->makeLeaf(rightPred);
        }
    } else {
        double rightSum = 0.0, rightWeight = 0.0;
        for (size_t i = 0; i < rightIndices_.size(); ++i) {
            double weight = (i < rightWeights.size()) ? rightWeights[i] : 1.0;
            rightSum += targets[rightIndices_[i]] * weight;
            rightWeight += weight;
        }
        double rightPred = (rightWeight > 0) ? rightSum / rightWeight : 0.0;
        leafInfo.node->rightChild->makeLeaf(rightPred);
    }
}