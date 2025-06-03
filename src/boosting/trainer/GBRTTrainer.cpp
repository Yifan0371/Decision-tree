// =============================================================================
// src/boosting/trainer/GBRTTrainer.cpp - OpenMP深度并行优化版本
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
#ifdef _OPENMP
#include <omp.h>
#endif

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
    
    #ifdef _OPENMP
    if (config_.verbose) {
        std::cout << "GBRT initialized with OpenMP support (" 
                  << omp_get_max_threads() << " threads)" << std::endl;
    }
    #endif
}

void GBRTTrainer::train(const std::vector<double>& X,
                       int rowLength,
                       const std::vector<double>& y) {
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    if (config_.enableDart) {
        trainWithDart(X, rowLength, y);
    } else {
        trainStandard(X, rowLength, y);
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    if (config_.verbose) {
        std::cout << "GBRT training completed in " << totalTime.count() 
                  << "ms with " << model_.getTreeCount() << " trees" << std::endl;
        
        #ifdef _OPENMP
        std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(y.size() * config_.numIterations) 
                      / (totalTime.count() * omp_get_max_threads())) 
                  << " samples/(ms*thread)" << std::endl;
        #endif
    }
}

void GBRTTrainer::trainStandard(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training standard GBRT with " << config_.numIterations 
                  << " iterations..." << std::endl;
    }
    
    size_t n = y.size();
    
    // **并行优化1: 基准分数计算的并行**
    double baseScore = computeBaseScore(y);
    model_.setBaseScore(baseScore);
    
    // 初始化预测值
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);  // 复用的树预测缓冲区
    
    trainingLoss_.reserve(config_.numIterations);
    
    // 标准Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        // **并行优化2: 当前损失计算的并行**
        double currentLoss = strategy_->computeTotalLoss(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // **并行优化3: 残差计算的并行（已在strategy中优化）**
        strategy_->updateTargets(y, currentPred, residuals);
        
        // 训练新树（内部已并行优化）
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // **并行优化4: 批量树预测的并行**
        batchTreePredict(treeTrainer.get(), X, rowLength, treePred);
        
        // 计算学习率
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // **并行优化5: 预测更新的并行（已在strategy中优化）**
        strategy_->updatePredictions(treePred, lr, currentPred);
        
        // 添加树到模型
        auto rootCopy = cloneTree(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 20 == 0) {
            std::cout << "Iter " << iter << " | Loss: " << std::fixed 
                      << std::setprecision(6) << currentLoss 
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
        // **并行优化6: 早停检查的优化**
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "Early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
}

void GBRTTrainer::trainWithDart(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training GBRT with DART (" << config_.numIterations 
                  << " iterations, drop rate: " << config_.dartDropRate << ")..." << std::endl;
    }
    
    size_t n = y.size();
    
    // 计算基准分数
    double baseScore = computeBaseScore(y);
    model_.setBaseScore(baseScore);
    
    // 初始化预测值和缓冲区
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);
    
    trainingLoss_.reserve(config_.numIterations);
    
    // DART Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        // 1. 选择要丢弃的树
        std::vector<int> droppedTrees;
        if (model_.getTreeCount() > 0) {
            droppedTrees = dartStrategy_->selectDroppedTrees(
                static_cast<int>(model_.getTreeCount()), 
                config_.dartDropRate, 
                dartGen_);
        }
        
        // **并行优化7: DART预测重计算的并行**
        updatePredictionsWithDropoutParallel(X, rowLength, droppedTrees, currentPred);
        
        // 计算当前损失
        double currentLoss = strategy_->computeTotalLoss(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // 计算残差
        strategy_->updateTargets(y, currentPred, residuals);
        
        // 训练新树
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // 获取新树预测
        batchTreePredict(treeTrainer.get(), X, rowLength, treePred);
        
        // 计算学习率
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // 添加新树到模型
        auto rootCopy = cloneTree(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        // DART权重更新
        int newTreeIndex = static_cast<int>(model_.getTreeCount()) - 1;
        dartStrategy_->updateTreeWeights(model_.getTrees(), droppedTrees, newTreeIndex, lr);
        
        // **并行优化8: 完整预测重计算的并行**
        batchModelPredict(X, rowLength, currentPred);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 20 == 0) {
            std::cout << "DART Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Dropped: " << droppedTrees.size() << " trees"
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
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
}

// **新增方法：批量树预测的并行版本**
void GBRTTrainer::batchTreePredict(const SingleTreeTrainer* trainer,
                                   const std::vector<double>& X,
                                   int rowLength,
                                   std::vector<double>& predictions) const {
    size_t n = predictions.size();
    
    // 批量预测的并行化
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = trainer->predict(&X[i * rowLength], rowLength);
    }
}

// **新增方法：批量模型预测的并行版本**
void GBRTTrainer::batchModelPredict(const std::vector<double>& X,
                                    int rowLength,
                                    std::vector<double>& predictions) const {
    size_t n = predictions.size();
    
    // 重置为基准分数
    std::fill(predictions.begin(), predictions.end(), model_.getBaseScore());
    
    // 并行累积所有树的预测
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = model_.predict(&X[i * rowLength], rowLength);
    }
}

// **优化版本：DART dropout预测重计算的并行**
void GBRTTrainer::updatePredictionsWithDropoutParallel(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<int>& droppedTrees,
    std::vector<double>& predictions) const {
    
    size_t n = predictions.size();
    
    // **并行优化9: DART预测重计算的高效并行**
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &X[i * rowLength];
        predictions[i] = dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), droppedTrees, sample, rowLength, model_.getBaseScore());
    }
}

std::vector<double> GBRTTrainer::predictBatch(
    const std::vector<double>& X, int rowLength) const {
    
    size_t n = X.size() / rowLength;
    std::vector<double> predictions;
    predictions.reserve(n);
    
    if (config_.enableDart && dartStrategy_ && config_.dartDropRate > 0.0) {
        // DART预测模式
        predictions.resize(n);
        #pragma omp parallel for schedule(static, 256) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
        }
    } else {
        // 标准预测模式（已在model中并行优化）
        predictions = model_.predictBatch(X, rowLength);
    }
    
    return predictions;
}

void GBRTTrainer::evaluate(const std::vector<double>& X,
                          int rowLength,
                          const std::vector<double>& y,
                          double& loss,
                          double& mse,
                          double& mae) {
    auto predictions = predictBatch(X, rowLength);
    size_t n = y.size();
    
    // **并行优化10: 评估指标计算的并行**
    loss = strategy_->computeTotalLoss(y, predictions);
    
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

double GBRTTrainer::computeBaseScore(const std::vector<double>& y) const {
    size_t n = y.size();
    double sum = 0.0;
    
    // **并行优化11: 基准分数计算的并行**
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

bool GBRTTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    
    // **并行优化12: 早停检查的向量化**
    auto recentStart = losses.end() - patience - 1;
    auto recentEnd = losses.end() - 1;
    
    double bestLoss = *std::min_element(recentStart, recentEnd);
    double currentLoss = losses.back();
    
    return currentLoss >= bestLoss - config_.tolerance;
}

// 保留原有方法...
std::unique_ptr<SingleTreeTrainer> GBRTTrainer::createTreeTrainer() const {
    auto criterion = std::make_unique<MSECriterion>();
    auto finder = std::make_unique<ExhaustiveSplitFinder>();
    auto pruner = std::make_unique<NoPruner>();
    
    return std::make_unique<SingleTreeTrainer>(
        std::move(finder), std::move(criterion), std::move(pruner),
        config_.maxDepth, config_.minSamplesLeaf);
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

std::unique_ptr<IDartStrategy> GBRTTrainer::createDartStrategy() const {
    if (config_.dartStrategy == "uniform") {
        return std::make_unique<UniformDartStrategy>(
            config_.dartNormalize, 
            config_.dartSkipDropForPrediction);
    } else {
        throw std::invalid_argument("Unsupported DART strategy: " + config_.dartStrategy);
    }
}

double GBRTTrainer::predict(const double* sample, int rowLength) const {
    if (config_.enableDart && dartStrategy_ && config_.dartDropRate > 0.0) {
        return dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
    } else {
        return model_.predict(sample, rowLength);
    }
}

void GBRTTrainer::updatePredictionsWithDropout(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<int>& droppedTrees,
    std::vector<double>& predictions) const {
    
    // 调用并行版本
    updatePredictionsWithDropoutParallel(X, rowLength, droppedTrees, predictions);
}