// =============================================================================
// src/boosting/trainer/GBRTTrainer.cpp (完整版本 - 支持DART)
// =============================================================================
#include "boosting/trainer/GBRTTrainer.hpp"
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"
#include "pruner/NoPruner.hpp"
#include "boosting/dart/UniformDartStrategy.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

GBRTTrainer::GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy)
    : config_(config), strategy_(std::move(strategy)), dartGen_(config.dartSeed) {
    
    // 如果启用DART，创建DART策略
    if (config_.enableDart) {
        dartStrategy_ = createDartStrategy();
        if (config_.verbose) {
            std::cout << "DART enabled with strategy: " << dartStrategy_->name() 
                      << ", drop rate: " << config_.dartDropRate << std::endl;
        }
    }
}

void GBRTTrainer::train(const std::vector<double>& X,
                       int rowLength,
                       const std::vector<double>& y) {
    
    if (config_.enableDart) {
        // 使用DART训练
        trainWithDart(X, rowLength, y);
    } else {
        // 使用标准GBRT训练
        trainStandard(X, rowLength, y);
    }
}

void GBRTTrainer::trainStandard(const std::vector<double>& X,
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
    
    // 标准Boosting迭代
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

void GBRTTrainer::trainWithDart(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training GBRT with DART (" << config_.numIterations 
                  << " iterations, drop rate: " << config_.dartDropRate << ")..." << std::endl;
        std::cout << "DEBUG: DART dropout rate = " << config_.dartDropRate << std::endl;
        std::cout << "DEBUG: DART normalize = " << config_.dartNormalize << std::endl;
    }
    
    size_t n = y.size();
    
    // 计算基准分数
    double baseScore = computeBaseScore(y);
    model_.setBaseScore(baseScore);
    
    // 初始化预测值
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    
    trainingLoss_.reserve(config_.numIterations);
    
    // DART Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        // 1. 选择要丢弃的树
        std::vector<int> droppedTrees;
        if (model_.getTreeCount() > 0) {
            droppedTrees = dartStrategy_->selectDroppedTrees(
                static_cast<int>(model_.getTreeCount()), 
                config_.dartDropRate, 
                dartGen_);
        }
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "DEBUG: Iter " << iter << " | Trees: " << model_.getTreeCount() 
                      << " | Dropped: " << droppedTrees.size() << std::endl;
        }
        
        // 2. 计算排除丢弃树的预测值
        updatePredictionsWithDropout(X, rowLength, droppedTrees, currentPred);
        
        // 3. 计算当前损失
        double currentLoss = strategy_->computeTotalLoss(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // 4. 计算残差（基于dropout预测）
        strategy_->updateTargets(y, currentPred, residuals);
        
        // 5. 训练新树
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // 6. 获取新树预测
        std::vector<double> treePred(n);
        for (size_t i = 0; i < n; ++i) {
            treePred[i] = treeTrainer->predict(&X[i * rowLength], rowLength);
        }
        
        // 7. 计算学习率
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // DEBUG: 输出学习率
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "DEBUG: Learning rate = " << lr << std::endl;
        }
        
        // 8. 添加新树到模型
        auto rootCopy = cloneTree(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        // 9. DART权重更新
        int newTreeIndex = static_cast<int>(model_.getTreeCount()) - 1;
        dartStrategy_->updateTreeWeights(model_.getTrees(), droppedTrees, newTreeIndex, lr);
        
        // 10. 更新完整预测（用于下一轮）- 重新计算避免累积误差
        for (size_t i = 0; i < n; ++i) {
            currentPred[i] = model_.predict(&X[i * rowLength], rowLength);
        }
        
        if (config_.verbose && iter % 20 == 0) {
            std::cout << "DART Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Dropped: " << droppedTrees.size() << " trees" << std::endl;
        }
        
        // 早停检查
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "Early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
    
    if (config_.verbose) {
        std::cout << "DART training completed: " << model_.getTreeCount() 
                  << " trees" << std::endl;
    }
}

double GBRTTrainer::predict(const double* sample, int rowLength) const {
    // 只有在实际有dropout时才使用DART预测路径
    if (config_.enableDart && dartStrategy_ && config_.dartDropRate > 0.0) {
        return dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
    } else {
        // 使用标准预测路径
        return model_.predict(sample, rowLength);
    }
}

std::vector<double> GBRTTrainer::predictBatch(
    const std::vector<double>& X, int rowLength) const {
    
    if (config_.enableDart && dartStrategy_ && config_.dartDropRate > 0.0) {
        // 只有在实际启用dropout时才使用DART预测
        size_t n = X.size() / rowLength;
        std::vector<double> predictions(n);
        
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
        }
        return predictions;
    } else {
        // 否则使用标准预测
        return model_.predictBatch(X, rowLength);
    }
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

// === DART专用方法实现 ===

std::unique_ptr<IDartStrategy> GBRTTrainer::createDartStrategy() const {
    if (config_.dartStrategy == "uniform") {
        return std::make_unique<UniformDartStrategy>(
            config_.dartNormalize, 
            config_.dartSkipDropForPrediction);
    } else {
        throw std::invalid_argument("Unsupported DART strategy: " + config_.dartStrategy);
    }
}

void GBRTTrainer::updatePredictionsWithDropout(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<int>& droppedTrees,
    std::vector<double>& predictions) const {
    
    size_t n = predictions.size();
    
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &X[i * rowLength];
        predictions[i] = dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), droppedTrees, sample, rowLength, model_.getBaseScore());
    }
}