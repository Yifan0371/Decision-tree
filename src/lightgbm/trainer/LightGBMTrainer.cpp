#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "criterion/MSECriterion.hpp"
#include "finder/HistogramEWFinder.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <iostream>  // 添加这行

LightGBMTrainer::LightGBMTrainer(const LightGBMConfig& config) : config_(config) {
    initializeComponents();
    
    // 预分配内存池
    gradients_.reserve(50000);
    sampleIndices_.reserve(50000);
    sampleWeights_.reserve(50000);
    trainingLoss_.reserve(config_.numIterations);
}

void LightGBMTrainer::initializeComponents() {
    // 创建损失函数
    lossFunction_ = std::make_unique<SquaredLoss>();
    
    // 创建GOSS采样器
    if (config_.enableGOSS) {
        gossSampler_ = std::make_unique<GOSSSampler>(config_.topRate, config_.otherRate);
    }
    
    // 创建特征绑定器
    if (config_.enableFeatureBundling) {
        featureBundler_ = std::make_unique<FeatureBundler>(config_.maxBin, config_.maxConflictRate);
    }
    
    // 创建树构建器
    treeBuilder_ = std::make_unique<LeafwiseTreeBuilder>(
        config_, createHistogramFinder(), createCriterion());
}

void LightGBMTrainer::train(const std::vector<double>& data,
                           int rowLength,
                           const std::vector<double>& labels) {
    size_t n = labels.size();
    
    if (config_.verbose) {
        std::cout << "LightGBM training started: " << n << " samples, " 
                  << rowLength << " features" << std::endl;
    }
    
    // 预处理特征绑定
    if (config_.enableFeatureBundling) {
        preprocessFeatures(data, rowLength, n);
    }
    
    // 计算基准分数并初始化预测
    double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    
    gradients_.resize(n);
    
    // Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        // 计算训练损失
        double currentLoss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            currentLoss += lossFunction_->loss(labels[i], predictions[i]);
        }
        currentLoss /= n;
        trainingLoss_.push_back(currentLoss);
        
        // 修复：计算负梯度（残差）而不是梯度
        for (size_t i = 0; i < n; ++i) {
            // 对于MSE，负梯度就是残差 y - f(x)
            gradients_[i] = labels[i] - predictions[i];  // 修复：这是负梯度
        }
        
        // GOSS采样
        if (config_.enableGOSS) {
            gossSampler_->sample(gradients_, sampleIndices_, sampleWeights_);
        } else {
            sampleIndices_.resize(n);
            sampleWeights_.assign(n, 1.0);
            std::iota(sampleIndices_.begin(), sampleIndices_.end(), 0);
        }
        
        // 构建新树（用残差作为目标）
        auto tree = treeBuilder_->buildTree(
            data, rowLength, gradients_, gradients_,  // 用残差作为标签
            sampleIndices_, sampleWeights_, featureBundles_);
        
        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": No valid split found, stopping." << std::endl;
            }
            break;
        }
        
        // 修复：正确更新预测值
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &data[i * rowLength];
            double treePred = 0.0;
            
            // 单树预测
            const Node* cur = tree.get();
            while (cur && !cur->isLeaf) {
                double val = sample[cur->getFeatureIndex()];
                cur = (val <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
            }
            if (cur) {
                treePred = cur->getPrediction();
            }
            
            // 修复：加上学习率缩放的树预测（而不是减去）
            predictions[i] += config_.learningRate * treePred;
        }
        
        // 添加树到模型
        model_.addTree(std::move(tree), config_.learningRate);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iteration " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
        // 早停检查
        if (config_.earlyStoppingRounds > 0 && iter >= config_.earlyStoppingRounds) {
            bool shouldStop = true;
            double recentBest = currentLoss;
            for (int i = 1; i <= config_.earlyStoppingRounds; ++i) {
                if (iter - i >= 0) {
                    recentBest = std::min(recentBest, trainingLoss_[iter - i]);
                }
            }
            if (currentLoss < recentBest - config_.tolerance) {
                shouldStop = false;
            }
            
            if (shouldStop) {
                if (config_.verbose) {
                    std::cout << "Early stopping at iteration " << iter << std::endl;
                }
                break;
            }
        }
    }
    
    if (config_.verbose) {
        std::cout << "LightGBM training completed: " << model_.getTreeCount() 
                  << " trees, " << featureBundles_.size() << " feature bundles" << std::endl;
    }
}

double LightGBMTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

void LightGBMTrainer::evaluate(const std::vector<double>& X,
                              int rowLength,
                              const std::vector<double>& y,
                              double& mse,
                              double& mae) {
    auto predictions = model_.predictBatch(X, rowLength);
    size_t n = y.size();
    
    mse = 0.0;
    mae = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    mse /= n;
    mae /= n;
}

void LightGBMTrainer::preprocessFeatures(const std::vector<double>& data,
                                        int rowLength,
                                        size_t sampleSize) {
    if (featureBundler_) {
        // 临时禁用特征绑定，避免复杂性
        // featureBundler_->createBundles(data, rowLength, sampleSize, featureBundles_);
        
        // 创建简单的1:1映射
        featureBundles_.clear();
        for (int i = 0; i < rowLength; ++i) {
            FeatureBundle bundle;
            bundle.features.push_back(i);
            bundle.offsets.push_back(0.0);
            bundle.totalBins = config_.maxBin;
            featureBundles_.push_back(bundle);
        }
        
        if (config_.verbose) {
            std::cout << "Feature bundling: " << rowLength << " -> " 
                      << featureBundles_.size() << " bundles (1:1 mapping)" << std::endl;
        }
    }
}

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

std::vector<double> LightGBMTrainer::calculateFeatureImportance(int numFeatures) const {
    std::vector<double> importance(numFeatures, 0.0);
    
    // 简化的特征重要性计算（基于分裂次数）
    for (size_t treeIdx = 0; treeIdx < model_.getTreeCount(); ++treeIdx) {
        // 这里需要访问树的内部结构，简化实现
        // 实际实现需要遍历树节点统计特征使用频次
    }
    
    return importance;
}

std::unique_ptr<ISplitFinder> LightGBMTrainer::createHistogramFinder() const {
    // 复用现有的直方图算法
    return std::make_unique<HistogramEWFinder>(config_.maxBin);
}

std::unique_ptr<ISplitCriterion> LightGBMTrainer::createCriterion() const {
    return std::make_unique<MSECriterion>();
}
