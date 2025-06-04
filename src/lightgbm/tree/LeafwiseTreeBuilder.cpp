// =============================================================================
// src/lightgbm/tree/LeafwiseTreeBuilder.cpp
// OpenMP 深度并行优化版本（减少锁竞争、提高阈值、预分配缓冲）
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
    const std::vector<double>& targets,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights,
    const std::vector<FeatureBundle>& /* bundles */) {

    // 清空优先队列
    while (!leafQueue_.empty()) leafQueue_.pop();

    // 初始化根节点
    auto root = std::make_unique<Node>();
    root->samples = sampleIndices.size();

    // 计算根节点预测（加权平均），阈值 n>=2000 并行
    double weightedSum = 0.0, totalWeight = 0.0;
    size_t n = sampleIndices.size();
    if (n >= 2000) {
        #pragma omp parallel for reduction(+:weightedSum, totalWeight) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double w = sampleWeights[i];
            weightedSum += targets[idx] * w;
            totalWeight += w;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double w = sampleWeights[i];
            weightedSum += targets[idx] * w;
            totalWeight += w;
        }
    }
    double rootPrediction = (totalWeight > 0.0) ? (weightedSum / totalWeight) : 0.0;

    // 根节点尝试分裂
    LeafInfo rootInfo;
    rootInfo.node = root.get();
    rootInfo.sampleIndices = sampleIndices;
    if (n < static_cast<size_t>(config_.minDataInLeaf) * 2) {
        // 样本太少，直接做叶
        root->makeLeaf(rootPrediction);
        return root;
    }
    if (n >= 2000) {
        if (!findBestSplitParallel(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
            root->makeLeaf(rootPrediction);
            return root;
        }
    } else {
        if (!findBestSplitSerial(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
            root->makeLeaf(rootPrediction);
            return root;
        }
    }
    leafQueue_.push(rootInfo);

    int currentLeaves = 1;
    while (!leafQueue_.empty() && currentLeaves < config_.numLeaves) {
        // 取增益最大的叶子
        LeafInfo bestLeaf = leafQueue_.top();
        leafQueue_.pop();

        // 如果不足以继续分裂
        size_t m = bestLeaf.sampleIndices.size();
        if (bestLeaf.splitGain <= config_.minSplitGain ||
            m < static_cast<size_t>(config_.minDataInLeaf) * 2) {
            // 直接并行/串行计算叶子预测，并置为叶子
            double leafPred = (m >= 500)
                              ? computeLeafPredictionParallel(bestLeaf.sampleIndices, targets, sampleWeights)
                              : computeLeafPredictionSerial(bestLeaf.sampleIndices, targets, sampleWeights);
            bestLeaf.node->makeLeaf(leafPred);
            continue;
        }

        // 并行或串行执行分裂
        if (m >= 2000) {
            splitLeafParallel(bestLeaf, data, rowLength, targets, sampleWeights);
        } else {
            splitLeafSerial(bestLeaf, data, rowLength, targets, sampleWeights);
        }
        currentLeaves++;
    }

    // 处理剩余所有节点：串行或并行
    if (!leafQueue_.empty()) {
        if (leafQueue_.size() >= 4) {
            processRemainingLeavesParallel(targets, sampleWeights);
        } else {
            processRemainingLeavesSerial(targets, sampleWeights);
        }
    }

    return root;
}

// 串行查找最佳 split
bool LeafwiseTreeBuilder::findBestSplitSerial(const std::vector<double>& data,
                                              int rowLength,
                                              const std::vector<double>& targets,
                                              const std::vector<int>& indices,
                                              const std::vector<double>& weights,
                                              LeafInfo& leafInfo) {
    if (indices.size() < static_cast<size_t>(config_.minDataInLeaf) * 2) return false;
    double currentMetric = criterion_->nodeMetric(targets, indices);
    auto [f, thresh, gain] =
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    leafInfo.bestFeature = f;
    leafInfo.bestThreshold = thresh;
    leafInfo.splitGain = gain;
    return f >= 0 && gain > 0;
}

// 并行查找最佳 split（本质与串行相同，仅称为 parallel）
bool LeafwiseTreeBuilder::findBestSplitParallel(const std::vector<double>& data,
                                                int rowLength,
                                                const std::vector<double>& targets,
                                                const std::vector<int>& indices,
                                                const std::vector<double>& weights,
                                                LeafInfo& leafInfo) {
    if (indices.size() < static_cast<size_t>(config_.minDataInLeaf) * 2) return false;
    double currentMetric = criterion_->nodeMetric(targets, indices);
    auto [f, thresh, gain] =
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    leafInfo.bestFeature = f;
    leafInfo.bestThreshold = thresh;
    leafInfo.splitGain = gain;
    return f >= 0 && gain > 0;
}

// 串行分裂叶子
void LeafwiseTreeBuilder::splitLeafSerial(LeafInfo& leafInfo,
                                          const std::vector<double>& data,
                                          int rowLength,
                                          const std::vector<double>& targets,
                                          const std::vector<double>& sampleWeights) {
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();

    leftIndices_.clear();
    rightIndices_.clear();
    leftWeights_.clear();
    rightWeights_.clear();

    for (size_t i = 0; i < leafInfo.sampleIndices.size(); ++i) {
        int idx = leafInfo.sampleIndices[i];
        double value = data[idx * rowLength + leafInfo.bestFeature];
        double w = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
        if (value <= leafInfo.bestThreshold) {
            leftIndices_.push_back(idx);
            leftWeights_.push_back(w);
        } else {
            rightIndices_.push_back(idx);
            rightWeights_.push_back(w);
        }
    }

    // 左子节点
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitSerial(data, rowLength, targets, leftInfo.sampleIndices, leftWeights_, leftInfo)) {
            leafQueue_.push(leftInfo);
        } else {
            double leftPred = computeLeafPredictionSerial(leftIndices_, targets, leftWeights_);
            leftInfo.node->makeLeaf(leftPred);
        }
    } else {
        double leftPred = computeLeafPredictionSerial(leftIndices_, targets, leftWeights_);
        leafInfo.node->leftChild->makeLeaf(leftPred);
    }

    // 右子节点（逻辑同左）
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitSerial(data, rowLength, targets, rightInfo.sampleIndices, rightWeights_, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightPred = computeLeafPredictionSerial(rightIndices_, targets, rightWeights_);
            rightInfo.node->makeLeaf(rightPred);
        }
    } else {
        double rightPred = computeLeafPredictionSerial(rightIndices_, targets, rightWeights_);
        leafInfo.node->rightChild->makeLeaf(rightPred);
    }
}

// 并行分裂叶子，减少锁竞争：用线程本地缓冲收集子节点，最后串行合并
void LeafwiseTreeBuilder::splitLeafParallel(LeafInfo& leafInfo,
                                            const std::vector<double>& data,
                                            int rowLength,
                                            const std::vector<double>& targets,
                                            const std::vector<double>& sampleWeights) {
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();

    size_t m = leafInfo.sampleIndices.size();
    leftIndices_.clear();
    rightIndices_.clear();
    leftWeights_.clear();
    rightWeights_.clear();
    leftIndices_.reserve(m / 2 + 1);
    rightIndices_.reserve(m / 2 + 1);
    leftWeights_.reserve(m / 2 + 1);
    rightWeights_.reserve(m / 2 + 1);

    // 并行判断放到局部缓冲
    int bestFeat = leafInfo.bestFeature;
    double bestThresh = leafInfo.bestThreshold;
    #pragma omp parallel
    {
        std::vector<int> localLeftIdx, localRightIdx;
        std::vector<double> localLeftW, localRightW;
        localLeftIdx.reserve(m / 4 + 1);
        localRightIdx.reserve(m / 4 + 1);
        localLeftW.reserve(m / 4 + 1);
        localRightW.reserve(m / 4 + 1);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < m; ++i) {
            int idx = leafInfo.sampleIndices[i];
            double value = data[idx * rowLength + bestFeat];
            double w = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
            if (value <= bestThresh) {
                localLeftIdx.push_back(idx);
                localLeftW.push_back(w);
            } else {
                localRightIdx.push_back(idx);
                localRightW.push_back(w);
            }
        }

        #pragma omp critical
        {
            leftIndices_.insert(leftIndices_.end(), localLeftIdx.begin(), localLeftIdx.end());
            leftWeights_.insert(leftWeights_.end(), localLeftW.begin(), localLeftW.end());
            rightIndices_.insert(rightIndices_.end(), localRightIdx.begin(), localRightIdx.end());
            rightWeights_.insert(rightWeights_.end(), localRightW.begin(), localRightW.end());
        }
    }

    // 左子节点
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitParallel(data, rowLength, targets, leftInfo.sampleIndices, leftWeights_, leftInfo)) {
            // 线程安全插入：这里只在并行段外，因为 findBestSplitParallel 只有判断，无插入冲突
            leafQueue_.push(leftInfo);
        } else {
            double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights_);
            leftInfo.node->makeLeaf(leftPred);
        }
    } else {
        double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights_);
        leafInfo.node->leftChild->makeLeaf(leftPred);
    }

    // 右子节点
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitParallel(data, rowLength, targets, rightInfo.sampleIndices, rightWeights_, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights_);
            rightInfo.node->makeLeaf(rightPred);
        }
    } else {
        double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights_);
        leafInfo.node->rightChild->makeLeaf(rightPred);
    }
}

double LeafwiseTreeBuilder::computeLeafPredictionSerial(
    const std::vector<int>& indices,
    const std::vector<double>& targets,
    const std::vector<double>& weights) const {
    if (indices.empty()) return 0.0;
    double sum = 0.0, wsum = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        sum += targets[indices[i]] * weights[i];
        wsum += weights[i];
    }
    return (wsum > 0.0) ? (sum / wsum) : 0.0;
}

double LeafwiseTreeBuilder::computeLeafPredictionParallel(
    const std::vector<int>& indices,
    const std::vector<double>& targets,
    const std::vector<double>& weights) const {
    if (indices.empty()) return 0.0;
    double sum = 0.0, wsum = 0.0;
    size_t m = indices.size();
    if (m >= 1000) {
        #pragma omp parallel for reduction(+:sum, wsum) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            sum += targets[indices[i]] * weights[i];
            wsum += weights[i];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            sum += targets[indices[i]] * weights[i];
            wsum += weights[i];
        }
    }
    return (wsum > 0.0) ? (sum / wsum) : 0.0;
}

// 串行处理剩余叶子
void LeafwiseTreeBuilder::processRemainingLeavesSerial(
    const std::vector<double>& targets,
    const std::vector<double>& sampleWeights) {
    while (!leafQueue_.empty()) {
        LeafInfo leaf = leafQueue_.top();
        leafQueue_.pop();
        double leafPred = computeLeafPredictionSerial(leaf.sampleIndices, targets, sampleWeights);
        leaf.node->makeLeaf(leafPred);
    }
}

// 并行处理剩余叶子：用并行 for
void LeafwiseTreeBuilder::processRemainingLeavesParallel(
    const std::vector<double>& targets,
    const std::vector<double>& sampleWeights) {
    // 收集所有剩余叶子到临时数组
    std::vector<LeafInfo> rem;
    rem.reserve(leafQueue_.size());
    while (!leafQueue_.empty()) {
        rem.push_back(leafQueue_.top());
        leafQueue_.pop();
    }
    size_t m = rem.size();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        double leafPred = computeLeafPredictionParallel(rem[i].sampleIndices, targets, sampleWeights);
        rem[i].node->makeLeaf(leafPred);
    }
}
