// =============================================================================
// src/boosting/trainer/GBRTTrainer.cpp (简化版本)
// =============================================================================
#include "boosting/trainer/GBRTTrainer.hpp"
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"
#include "pruner/NoPruner.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

GBRTTrainer::GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy)
    : config_(config), strategy_(std::move(strategy)) {}

void GBRTTrainer::train(const std::vector<double>& X,
                       int rowLength,
                       const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training GBRT with " << config_.numIterations 
                  << " iterations..." << std::endl;
    }
    
    size_t n = y.size();
    
    // 计算基准分数
    double baseScore = computeBaseScore(y);
    model_.setBaseScore(baseScore);
    
    // 初始化预测值
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    
    trainingLoss_.reserve(config_.numIterations);
    
    // Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        // 计算当前损失
        double currentLoss = strategy_->computeTotalLoss(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // 计算残差
        strategy_->updateTargets(y, currentPred, residuals);
        
        // 训练新树
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // 获取树预测
        std::vector<double> treePred(n);
        for (size_t i = 0; i < n; ++i) {
            treePred[i] = treeTrainer->predict(&X[i * rowLength], rowLength);
        }
        
        // 计算学习率
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // 更新预测
        strategy_->updatePredictions(treePred, lr, currentPred);
        
        // 添加树到模型
        auto rootCopy = cloneTree(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        if (config_.verbose && iter % 20 == 0) {
            std::cout << "Iter " << iter << " | Loss: " << std::fixed 
                      << std::setprecision(6) << currentLoss << std::endl;
        }
    }
    
    if (config_.verbose) {
        std::cout << "Training completed: " << model_.getTreeCount() 
                  << " trees" << std::endl;
    }
}

double GBRTTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

std::vector<double> GBRTTrainer::predictBatch(
    const std::vector<double>& X, int rowLength) const {
    return model_.predictBatch(X, rowLength);
}

void GBRTTrainer::evaluate(const std::vector<double>& X,
                          int rowLength,
                          const std::vector<double>& y,
                          double& loss,
                          double& mse,
                          double& mae) {
    auto predictions = predictBatch(X, rowLength);
    size_t n = y.size();
    
    loss = strategy_->computeTotalLoss(y, predictions);
    
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

std::unique_ptr<SingleTreeTrainer> GBRTTrainer::createTreeTrainer() const {
    auto criterion = std::make_unique<MSECriterion>();
    auto finder = std::make_unique<ExhaustiveSplitFinder>();
    auto pruner = std::make_unique<NoPruner>();
    
    return std::make_unique<SingleTreeTrainer>(
        std::move(finder), std::move(criterion), std::move(pruner),
        config_.maxDepth, config_.minSamplesLeaf);
}

double GBRTTrainer::computeBaseScore(const std::vector<double>& y) const {
    return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

bool GBRTTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    
    double bestLoss = *std::min_element(losses.end() - patience - 1, losses.end() - 1);
    double currentLoss = losses.back();
    
    return currentLoss >= bestLoss - config_.tolerance;
}

void GBRTTrainer::sampleData(const std::vector<double>& /* X */, int /* rowLength */,
                            const std::vector<double>& /* residuals */,
                            std::vector<double>& /* sampledX */,
                            std::vector<double>& /* sampledResiduals */) const {
    // 暂时空实现
}

std::unique_ptr<Node> GBRTTrainer::cloneTree(const Node* original) const {
    if (!original) return nullptr;
    
    auto clone = std::make_unique<Node>();
    clone->isLeaf = original->isLeaf;
    clone->samples = original->samples;
    clone->metric = original->metric;
    
    if (original->isLeaf) {
        clone->makeLeaf(original->getPrediction(), original->getNodePrediction());
    } else {
        clone->makeInternal(original->getFeatureIndex(), original->getThreshold());
        clone->leftChild = cloneTree(original->getLeft());
        clone->rightChild = cloneTree(original->getRight());
        
        if (clone->leftChild) {
            clone->info.internal.left = clone->leftChild.get();
        }
        if (clone->rightChild) {
            clone->info.internal.right = clone->rightChild.get();
        }
    }
    
    return clone;
}

double GBRTTrainer::computeValidationLoss(const std::vector<double>& predictions) const {
    return strategy_->computeTotalLoss(y_val_, predictions);
}