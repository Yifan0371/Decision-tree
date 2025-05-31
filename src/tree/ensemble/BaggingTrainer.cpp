#include "ensemble/BaggingTrainer.hpp"

// 准则
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "criterion/HuberCriterion.hpp"
#include "criterion/QuantileCriterion.hpp"
#include "criterion/LogCoshCriterion.hpp"
#include "criterion/PoissonCriterion.hpp"

// 分割器
#include "finder/ExhaustiveSplitFinder.hpp"
#include "finder/RandomSplitFinder.hpp"
#include "finder/QuartileSplitFinder.hpp"
#include "finder/HistogramEWFinder.hpp"
#include "finder/HistogramEQFinder.hpp"
#include "finder/AdaptiveEWFinder.hpp"
#include "finder/AdaptiveEQFinder.hpp"

// 剪枝器
#include "pruner/NoPruner.hpp"
#include "pruner/MinGainPrePruner.hpp"
#include "pruner/CostComplexityPruner.hpp"
#include "pruner/ReducedErrorPruner.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <functional>

BaggingTrainer::BaggingTrainer(int numTrees,
                               double sampleRatio,
                               int maxDepth,
                               int minSamplesLeaf,
                               const std::string& criterion,
                               const std::string& splitMethod,
                               const std::string& prunerType,
                               double prunerParam,
                               uint32_t seed)
    : numTrees_(numTrees),
      sampleRatio_(sampleRatio),
      maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf),
      criterion_(criterion),
      splitMethod_(splitMethod),
      prunerType_(prunerType),
      prunerParam_(prunerParam),
      gen_(seed) {
    
    trees_.reserve(numTrees_);
    oobIndices_.reserve(numTrees_);
}

std::unique_ptr<ISplitFinder> BaggingTrainer::createSplitFinder() const {
    const std::string& method = splitMethod_;
    
    if (method == "exhaustive" || method == "exact") {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
    else if (method == "random" || method.find("random:") == 0) {
        int k = 10;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            k = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<RandomSplitFinder>(k);
    }
    else if (method == "quartile") {
        return std::make_unique<QuartileSplitFinder>();
    }
    else if (method == "histogram_ew" || method.find("histogram_ew:") == 0) {
        int bins = 64;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEWFinder>(bins);
    }
    else if (method == "histogram_eq" || method.find("histogram_eq:") == 0) {
        int bins = 64;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEQFinder>(bins);
    }
    else if (method == "adaptive_ew" || method.find("adaptive_ew:") == 0) {
        std::string rule = "sturges";
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            rule = method.substr(pos + 1);
        }
        return std::make_unique<AdaptiveEWFinder>(8, 128, rule);
    }
    else if (method == "adaptive_eq") {
        return std::make_unique<AdaptiveEQFinder>(5, 64, 0.1);
    }
    else {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
}

std::unique_ptr<ISplitCriterion> BaggingTrainer::createCriterion() const {
    const std::string& crit = criterion_;
    
    if (crit == "mae")
        return std::make_unique<MAECriterion>();
    else if (crit == "huber")
        return std::make_unique<HuberCriterion>();
    else if (crit.rfind("quantile", 0) == 0) {
        double tau = 0.5;
        auto pos = crit.find(':');
        if (pos != std::string::npos)
            tau = std::stod(crit.substr(pos + 1));
        return std::make_unique<QuantileCriterion>(tau);
    }
    else if (crit == "logcosh")
        return std::make_unique<LogCoshCriterion>();
    else if (crit == "poisson")
        return std::make_unique<PoissonCriterion>();
    else
        return std::make_unique<MSECriterion>();
}

std::unique_ptr<IPruner> BaggingTrainer::createPruner(const std::vector<double>& X_val,
                                                     int rowLength,
                                                     const std::vector<double>& y_val) const {
    if (prunerType_ == "mingain") {
        return std::make_unique<MinGainPrePruner>(prunerParam_);
    }
    else if (prunerType_ == "cost_complexity") {
        return std::make_unique<CostComplexityPruner>(prunerParam_);
    }
    else if (prunerType_ == "reduced_error") {
        if (X_val.empty() || y_val.empty()) {
            return std::make_unique<NoPruner>();
        }
        return std::make_unique<ReducedErrorPruner>(X_val, rowLength, y_val);
    }
    else {
        return std::make_unique<NoPruner>();
    }
}

void BaggingTrainer::bootstrapSample(int dataSize,
                                     std::vector<int>& sampleIndices,
                                     std::vector<int>& oobIndices) const {
    int sampleSize = static_cast<int>(dataSize * sampleRatio_);
    
    sampleIndices.clear();
    sampleIndices.reserve(sampleSize);
    
    std::uniform_int_distribution<int> dist(0, dataSize - 1);
    std::unordered_set<int> sampledSet;
    
    // Bootstrap采样（有放回采样）
    for (int i = 0; i < sampleSize; ++i) {
        int idx = dist(gen_);
        sampleIndices.push_back(idx);
        sampledSet.insert(idx);
    }
    
    // 计算袋外样本
    oobIndices.clear();
    for (int i = 0; i < dataSize; ++i) {
        if (sampledSet.find(i) == sampledSet.end()) {
            oobIndices.push_back(i);
        }
    }
}

void BaggingTrainer::extractSubset(const std::vector<double>& originalData,
                                  int rowLength,
                                  const std::vector<double>& originalLabels,
                                  const std::vector<int>& indices,
                                  std::vector<double>& subData,
                                  std::vector<double>& subLabels) const {
    int featCount = rowLength; // rowLength 在这里应该是特征数，不包括标签
    
    subData.clear();
    subLabels.clear();
    subData.reserve(indices.size() * featCount);
    subLabels.reserve(indices.size());
    
    for (int idx : indices) {
        // 复制特征
        for (int f = 0; f < featCount; ++f) {
            subData.push_back(originalData[idx * featCount + f]);
        }
        // 复制标签
        subLabels.push_back(originalLabels[idx]);
    }
}

void BaggingTrainer::train(const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels) {
    trees_.clear();
    oobIndices_.clear();
    
    int dataSize = static_cast<int>(labels.size());
    std::cout << "Training " << numTrees_ << " trees with bagging..." << std::endl;
    
    for (int t = 0; t < numTrees_; ++t) {
        // Bootstrap采样
        std::vector<int> sampleIndices, oobIndices;
        bootstrapSample(dataSize, sampleIndices, oobIndices);
        oobIndices_.push_back(oobIndices);
        
        // 提取子数据集
        std::vector<double> subData, subLabels;
        extractSubset(data, rowLength, labels, sampleIndices, subData, subLabels);
        
        // 创建单个树的组件
        auto finder = createSplitFinder();
        auto criterion = createCriterion();
        auto pruner = createPruner({}, rowLength, {}); // Bagging通常不用验证集剪枝
        
        // 创建并训练树
        auto tree = std::make_unique<SingleTreeTrainer>(
            std::move(finder),
            std::move(criterion), 
            std::move(pruner),
            maxDepth_,
            minSamplesLeaf_
        );
        
        tree->train(subData, rowLength, subLabels);
        trees_.push_back(std::move(tree));
        
        // 进度输出
        if ((t + 1) % std::max(1, numTrees_ / 10) == 0) {
            std::cout << "Completed " << (t + 1) << "/" << numTrees_ 
                      << " trees (" << std::fixed << std::setprecision(1) 
                      << 100.0 * (t + 1) / numTrees_ << "%)" << std::endl;
        }
    }
    
    std::cout << "Bagging training completed!" << std::endl;
}

double BaggingTrainer::predict(const double* sample, int rowLength) const {
    if (trees_.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& tree : trees_) {
        sum += tree->predict(sample, rowLength);
    }
    return sum / trees_.size();
}

void BaggingTrainer::evaluate(const std::vector<double>& X,
                             int rowLength,
                             const std::vector<double>& y,
                             double& mse,
                             double& mae) {
    size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double pred = predict(&X[i * rowLength], rowLength);
        double diff = y[i] - pred;
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

std::vector<double> BaggingTrainer::getFeatureImportance(int numFeatures) const {
    std::vector<double> importance(numFeatures, 0.0);
    
    // 遍历所有树计算特征重要性
    for (const auto& tree : trees_) {
        const Node* root = tree->getRoot();
        if (!root || root->isLeaf) continue;
        
        // 使用栈实现的非递归遍历
        std::vector<const Node*> stack;
        stack.push_back(root);
        
        while (!stack.empty()) {
            const Node* node = stack.back();
            stack.pop_back();
            
            if (!node || node->isLeaf) continue;
            
            int feat = node->getFeatureIndex();
            if (feat >= 0 && feat < numFeatures) {
                importance[feat] += 1.0;
            }
            
            // 添加子节点到栈中
            if (node->getLeft()) stack.push_back(node->getLeft());
            if (node->getRight()) stack.push_back(node->getRight());
        }
    }
    
    // 归一化
    double total = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (total > 0) {
        for (double& imp : importance) {
            imp /= total;
        }
    }
    
    return importance;
}

double BaggingTrainer::getOOBError(const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels) const {
    if (trees_.empty() || oobIndices_.empty()) return 0.0;
    
    int dataSize = static_cast<int>(labels.size());
    std::vector<double> oobPredictions(dataSize, 0.0);
    std::vector<int> oobCounts(dataSize, 0);
    
    // 对每棵树，使用其OOB样本进行预测
    for (size_t t = 0; t < trees_.size(); ++t) {
        for (int idx : oobIndices_[t]) {
            double pred = trees_[t]->predict(&data[idx * rowLength], rowLength);
            oobPredictions[idx] += pred;
            oobCounts[idx]++;
        }
    }
    
    // 计算OOB误差
    double oobMSE = 0.0;
    int validCount = 0;
    
    for (int i = 0; i < dataSize; ++i) {
        if (oobCounts[i] > 0) {
            double avgPred = oobPredictions[i] / oobCounts[i];
            double diff = labels[i] - avgPred;
            oobMSE += diff * diff;
            validCount++;
        }
    }
    
    return validCount > 0 ? oobMSE / validCount : 0.0;
}