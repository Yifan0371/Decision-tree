#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include <algorithm>
#include <numeric>

std::unique_ptr<Node> LeafwiseTreeBuilder::buildTree(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& labels,  // 这个是原始标签
    const std::vector<double>& targets, // 这个是残差/梯度
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights,
    const std::vector<FeatureBundle>& /* bundles */) {
    
    // 清空优先队列
    while (!leafQueue_.empty()) leafQueue_.pop();
    
    // 创建根节点
    auto root = std::make_unique<Node>();
    root->samples = sampleIndices.size();
    
    // 修复：计算加权平均的残差作为根节点预测值
    double weightedSum = 0.0, totalWeight = 0.0;
    for (size_t i = 0; i < sampleIndices.size(); ++i) {
        int idx = sampleIndices[i];
        double weight = sampleWeights[i];
        weightedSum += targets[idx] * weight;  // 使用残差
        totalWeight += weight;
    }
    double rootPrediction = weightedSum / totalWeight;
    
    // 计算根节点的最佳分裂
    LeafInfo rootInfo;
    rootInfo.node = root.get();
    rootInfo.sampleIndices = sampleIndices;
    
    // 修复：使用残差进行分裂计算
    if (findBestSplit(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
        leafQueue_.push(rootInfo);
    } else {
        root->makeLeaf(rootPrediction);
        return root;
    }
    
    // Leaf-wise生长
    int numLeaves = 1;
    while (!leafQueue_.empty() && numLeaves < config_.numLeaves) {
        LeafInfo bestLeaf = leafQueue_.top();
        leafQueue_.pop();
        
        // 检查分裂条件
        if (bestLeaf.splitGain <= config_.minSplitGain ||
            bestLeaf.sampleIndices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
            
            // 计算叶子预测值（使用残差）
            double leafSum = 0.0, leafWeight = 0.0;
            for (size_t i = 0; i < bestLeaf.sampleIndices.size(); ++i) {
                int idx = bestLeaf.sampleIndices[i];
                leafSum += targets[idx] * sampleWeights[i];
                leafWeight += sampleWeights[i];
            }
            bestLeaf.node->makeLeaf(leafSum / leafWeight);
            continue;
        }
        
        // 执行分裂
        splitLeaf(bestLeaf, data, rowLength, targets, sampleWeights);
        numLeaves += 1;
    }
    
    // 将队列中剩余的节点设为叶子
    while (!leafQueue_.empty()) {
        LeafInfo remainingLeaf = leafQueue_.top();
        leafQueue_.pop();
        
        double leafSum = 0.0, leafWeight = 0.0;
        for (size_t i = 0; i < remainingLeaf.sampleIndices.size(); ++i) {
            int idx = remainingLeaf.sampleIndices[i];
            leafSum += targets[idx] * sampleWeights[i];
            leafWeight += sampleWeights[i];
        }
        remainingLeaf.node->makeLeaf(leafSum / leafWeight);
    }
    
    return root;
}

bool LeafwiseTreeBuilder::findBestSplit(const std::vector<double>& data,
                                       int rowLength,
                                       const std::vector<double>& labels,
                                       const std::vector<int>& indices,
                                       const std::vector<double>& /* weights */,  // 避免警告
                                       LeafInfo& leafInfo) {
    
    if (indices.size() < 2 * static_cast<size_t>(config_.minDataInLeaf)) {
        return false;
    }
    
    // 计算当前节点的metric
    double currentMetric = criterion_->nodeMetric(labels, indices);
    
    // 寻找最佳分裂
    auto [bestFeature, bestThreshold, bestGain] = 
        finder_->findBestSplit(data, rowLength, labels, indices, currentMetric, *criterion_);
    
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
    
    // 设置节点为内部节点
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    
    // 创建子节点
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();
    
    // 分割样本
    leftIndices_.clear();
    rightIndices_.clear();
    std::vector<double> leftWeights, rightWeights;
    
    for (size_t i = 0; i < leafInfo.sampleIndices.size(); ++i) {
        int idx = leafInfo.sampleIndices[i];
        double value = data[idx * rowLength + leafInfo.bestFeature];
        if (value <= leafInfo.bestThreshold) {
            leftIndices_.push_back(idx);
            leftWeights.push_back(sampleWeights[i]);
        } else {
            rightIndices_.push_back(idx);
            rightWeights.push_back(sampleWeights[i]);
        }
    }
    
    // 创建左右子节点的LeafInfo
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        
        if (findBestSplit(data, rowLength, targets, leftInfo.sampleIndices, leftWeights, leftInfo)) {
            leafQueue_.push(leftInfo);
        } else {
            double leftSum = 0.0, leftWeight = 0.0;
            for (size_t i = 0; i < leftIndices_.size(); ++i) {
                leftSum += targets[leftIndices_[i]] * leftWeights[i];
                leftWeight += leftWeights[i];
            }
            leftInfo.node->makeLeaf(leftSum / leftWeight);
        }
    }
    
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        
        std::vector<double> rightWeights2;
        for (size_t i = 0; i < rightIndices_.size(); ++i) {
            rightWeights2.push_back(sampleWeights[i]);
        }
        
        if (findBestSplit(data, rowLength, targets, rightInfo.sampleIndices, rightWeights2, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightSum = 0.0, rightWeight = 0.0;
            for (size_t i = 0; i < rightIndices_.size(); ++i) {
                rightSum += targets[rightIndices_[i]] * rightWeights2[i];
                rightWeight += rightWeights2[i];
            }
            rightInfo.node->makeLeaf(rightSum / rightWeight);
        }
    }
}
