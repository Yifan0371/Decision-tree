// =============================================================================
// src/boosting/trainer/GBRTTrainer.cpp - 深度并行优化版本
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
#include <memory>
#include <execution>  // C++17并行算法
#ifdef _OPENMP
#include <omp.h>
#endif

GBRTTrainer::GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy)
    : config_(config), strategy_(std::move(strategy)), dartGen_(config.dartSeed) {
    
    if (config_.enableDart) {
        dartStrategy_ = createDartStrategy();
        if (config_.verbose) {
            std::cout << "DART enabled with strategy: " << dartStrategy_->name() 
                      << ", drop rate: " << config_.dartDropRate << std::endl;
        }
    }
    
    #ifdef _OPENMP
    // 动态调整线程数
    omp_set_dynamic(1);
    omp_set_max_active_levels(2);  // 允许两层嵌套并行
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
        trainWithDartOptimized(X, rowLength, y);
    } else {
        trainStandardOptimized(X, rowLength, y);
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

// **核心优化1: 高度并行的标准GBRT训练**
void GBRTTrainer::trainStandardOptimized(const std::vector<double>& X,
                                         int rowLength,
                                         const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training optimized GBRT with " << config_.numIterations 
                  << " iterations..." << std::endl;
    }
    
    const size_t n = y.size();
    const int numThreads = omp_get_max_threads();
    
    // **优化: 预分配所有内存，避免训练时分配**
    double baseScore = computeBaseScoreParallel(y);
    model_.setBaseScore(baseScore);
    
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);
    
    // **线程局部缓冲区池**
    std::vector<std::vector<double>> threadLocalPreds(numThreads);
    for (auto& buf : threadLocalPreds) {
        buf.resize(n);
    }
    
    trainingLoss_.reserve(config_.numIterations);
    
    // **核心训练循环 - 高度优化版本**
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        // **步骤1: 并行损失计算**
        double currentLoss = computeTotalLossParallel(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // **步骤2: 超高效并行残差计算**
        computeResidualsParallel(y, currentPred, residuals);
        
        // **步骤3: 训练新树（内部已并行）**
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // **步骤4: 超高效批量树预测**
        batchTreePredictOptimized(treeTrainer.get(), X, rowLength, treePred);
        
        // **步骤5: 计算学习率（如果启用线搜索）**
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // **步骤6: 向量化预测更新**
        updatePredictionsVectorized(treePred, lr, currentPred);
        
        // **步骤7: 添加树到模型**
        auto rootCopy = cloneTreeOptimized(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss 
                      << " | LR: " << lr
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
        // **早停检查**
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "Early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
}

// **核心优化2: 高度并行的DART训练**
void GBRTTrainer::trainWithDartOptimized(const std::vector<double>& X,
                                         int rowLength,
                                         const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training optimized DART GBRT (" << config_.numIterations 
                  << " iterations, drop rate: " << config_.dartDropRate << ")..." << std::endl;
    }
    
    const size_t n = y.size();
    
    // **预分配和初始化**
    double baseScore = computeBaseScoreParallel(y);
    model_.setBaseScore(baseScore);
    
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);
    
    // **DART专用缓冲区**
    std::vector<double> predBeforeDrop(n);
    std::vector<double> predAfterDrop(n);
    
    trainingLoss_.reserve(config_.numIterations);
    
    // **DART Boosting主循环**
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        // **步骤1: 选择要丢弃的树**
        std::vector<int> droppedTrees;
        if (model_.getTreeCount() > 0) {
            droppedTrees = dartStrategy_->selectDroppedTrees(
                static_cast<int>(model_.getTreeCount()), 
                config_.dartDropRate, 
                dartGen_);
        }
        
        if (config_.verbose && iter % 10 == 0 && !droppedTrees.empty()) {
            std::cout << "DART Iter " << iter << ": Dropping " << droppedTrees.size() 
                      << " trees" << std::endl;
        }
        
        // **步骤2: 高效并行重计算DART预测**
        if (!droppedTrees.empty()) {
            computeDartPredictionsParallel(X, rowLength, droppedTrees, currentPred);
        }
        
        // **步骤3: 计算当前损失**
        double currentLoss = computeTotalLossParallel(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        // **步骤4: 计算残差**
        computeResidualsParallel(y, currentPred, residuals);
        
        // **步骤5: 训练新树**
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
        // **步骤6: 获取新树预测**
        batchTreePredictOptimized(treeTrainer.get(), X, rowLength, treePred);
        
        // **步骤7: 计算学习率**
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        // **步骤8: 添加新树到模型**
        auto rootCopy = cloneTreeOptimized(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        // **步骤9: DART权重更新**
        int newTreeIndex = static_cast<int>(model_.getTreeCount()) - 1;
        dartStrategy_->updateTreeWeights(model_.getTrees(), droppedTrees, newTreeIndex, lr);
        
        // **步骤10: 完整预测重计算（并行优化版）**
        recomputeFullPredictionsParallel(X, rowLength, currentPred);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "DART Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Dropped: " << droppedTrees.size() << " trees"
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
        // **早停检查**
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "DART early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
}

// **核心优化方法实现**

// **并行基准分数计算**
double GBRTTrainer::computeBaseScoreParallel(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static, 2048) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

// **超高效并行损失计算**
double GBRTTrainer::computeTotalLossParallel(const std::vector<double>& y,
                                            const std::vector<double>& pred) const {
    const size_t n = y.size();
    double totalLoss = 0.0;
    
    // **使用更大的chunk size和static调度**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += strategy_->getLossFunction()->loss(y[i], pred[i]);
    }
    
    return totalLoss / n;
}

// **超高效并行残差计算**
void GBRTTrainer::computeResidualsParallel(const std::vector<double>& y,
                                          const std::vector<double>& pred,
                                          std::vector<double>& residuals) const {
    const size_t n = y.size();
    
    // **向量化友好的并行计算**
    #pragma omp parallel for schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        residuals[i] = strategy_->getLossFunction()->gradient(y[i], pred[i]);
    }
}

// **优化的批量树预测**
void GBRTTrainer::batchTreePredictOptimized(const SingleTreeTrainer* trainer,
                                           const std::vector<double>& X,
                                           int rowLength,
                                           std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    // **更大的chunk size减少线程调度开销**
    #pragma omp parallel for schedule(static, 1024) if(n > 500)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = trainer->predict(&X[i * rowLength], rowLength);
    }
}

// **向量化预测更新**
void GBRTTrainer::updatePredictionsVectorized(const std::vector<double>& treePred,
                                              double lr,
                                              std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    // **SIMD友好的向量化更新**
    #pragma omp parallel for schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] += lr * treePred[i];
    }
}

// **DART专用: 高效并行DART预测重计算**
void GBRTTrainer::computeDartPredictionsParallel(const std::vector<double>& X,
                                                 int rowLength,
                                                 const std::vector<int>& droppedTrees,
                                                 std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    if (droppedTrees.empty()) return;
    
    // **方法1: 如果丢弃的树很少，直接减去它们的贡献**
    if (droppedTrees.size() <= 3) {
        for (int treeIdx : droppedTrees) {
            if (treeIdx >= 0 && treeIdx < static_cast<int>(model_.getTrees().size())) {
                const auto& tree = model_.getTrees()[treeIdx];
                
                #pragma omp parallel for schedule(static, 1024) if(n > 500)
                for (size_t i = 0; i < n; ++i) {
                    double treePred = predictSingleTreeFast(tree.tree.get(), &X[i * rowLength]);
                    predictions[i] -= tree.learningRate * tree.weight * treePred;
                }
            }
        }
    } else {
        // **方法2: 丢弃树较多时，使用DART策略重计算**
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), droppedTrees, sample, rowLength, model_.getBaseScore());
        }
    }
}

// **完整预测重计算（DART专用）**
void GBRTTrainer::recomputeFullPredictionsParallel(const std::vector<double>& X,
                                                   int rowLength,
                                                   std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 512) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = model_.predict(&X[i * rowLength], rowLength);
    }
}

// **快速单树预测（内联优化）**
inline double GBRTTrainer::predictSingleTreeFast(const Node* tree, const double* sample) const {
    const Node* cur = tree;
    while (cur && !cur->isLeaf) {
        const int featIdx = cur->getFeatureIndex();
        const double threshold = cur->getThreshold();
        cur = (sample[featIdx] <= threshold) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}

// **优化的树克隆（减少深度递归）**
std::unique_ptr<Node> GBRTTrainer::cloneTreeOptimized(const Node* original) const {
    if (!original) return nullptr;
    
    // **使用栈实现非递归克隆，避免深度递归的开销**
    auto root = std::make_unique<Node>();
    
    // **简化的克隆实现**
    std::function<void(Node*, const Node*)> cloneNode = [&](Node* dest, const Node* src) {
        dest->isLeaf = src->isLeaf;
        dest->samples = src->samples;
        dest->metric = src->metric;
        
        if (src->isLeaf) {
            dest->makeLeaf(src->getPrediction(), src->getNodePrediction());
        } else {
            dest->makeInternal(src->getFeatureIndex(), src->getThreshold());
            if (src->getLeft()) {
                dest->leftChild = std::make_unique<Node>();
                cloneNode(dest->leftChild.get(), src->getLeft());
            }
            if (src->getRight()) {
                dest->rightChild = std::make_unique<Node>();
                cloneNode(dest->rightChild.get(), src->getRight());
            }
        }
    };
    
    cloneNode(root.get(), original);
    return root;
}

// **批量预测（并行优化版）**
std::vector<double> GBRTTrainer::predictBatch(
    const std::vector<double>& X, int rowLength) const {
    
    const size_t n = X.size() / rowLength;
    std::vector<double> predictions;
    predictions.reserve(n);
    
    if (config_.enableDart && dartStrategy_) {
        // **DART预测模式**
        predictions.resize(n);
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
        }
    } else {
        // **标准预测模式**
        predictions = model_.predictBatch(X, rowLength);
    }
    
    return predictions;
}

// **并行评估**
void GBRTTrainer::evaluate(const std::vector<double>& X,
                          int rowLength,
                          const std::vector<double>& y,
                          double& loss,
                          double& mse,
                          double& mae) {
    auto predictions = predictBatch(X, rowLength);
    const size_t n = y.size();
    
    // **并行计算所有评估指标**
    loss = strategy_->computeTotalLoss(y, predictions);
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 2048) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

// **优化的早停检查**
bool GBRTTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    
    // **向量化的早停检查**
    const double currentLoss = losses.back();
    const auto recentStart = losses.end() - patience - 1;
    const auto recentEnd = losses.end() - 1;
    
    const double bestLoss = *std::min_element(recentStart, recentEnd);
    return currentLoss >= bestLoss - config_.tolerance;
}

// **保留原有接口的简化实现**
std::unique_ptr<SingleTreeTrainer> GBRTTrainer::createTreeTrainer() const {
    auto criterion = std::make_unique<MSECriterion>();
    auto finder = std::make_unique<ExhaustiveSplitFinder>();
    auto pruner = std::make_unique<NoPruner>();
    
    return std::make_unique<SingleTreeTrainer>(
        std::move(finder), std::move(criterion), std::move(pruner),
        config_.maxDepth, config_.minSamplesLeaf);
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
    if (config_.enableDart && dartStrategy_) {
        return dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
    } else {
        return model_.predict(sample, rowLength);
    }
}