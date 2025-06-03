// =============================================================================
// src/lightgbm/trainer/LightGBMTrainer.cpp - OpenMP深度并行优化版本
// =============================================================================
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "criterion/MSECriterion.hpp"

// 分割器includes
#include "finder/HistogramEWFinder.hpp"
#include "finder/HistogramEQFinder.hpp"
#include "finder/AdaptiveEWFinder.hpp"
#include "finder/AdaptiveEQFinder.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"

#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

LightGBMTrainer::LightGBMTrainer(const LightGBMConfig& config)
    : config_(config) {
    initializeComponents();

    // 预分配内存池
    gradients_.reserve(50000);
    sampleIndices_.reserve(50000);
    sampleWeights_.reserve(50000);
    trainingLoss_.reserve(config_.numIterations);
    
    #ifdef _OPENMP
    if (config_.verbose) {
        std::cout << "LightGBM initialized with OpenMP support (" 
                  << omp_get_max_threads() << " threads)" << std::endl;
    }
    #endif
}

void LightGBMTrainer::initializeComponents() {
    lossFunction_ = std::make_unique<SquaredLoss>();

    if (config_.enableGOSS) {
        gossSampler_ = std::make_unique<GOSSSampler>(config_.topRate, config_.otherRate);
    }

    if (config_.enableFeatureBundling) {
        featureBundler_ = std::make_unique<FeatureBundler>(config_.maxBin, config_.maxConflictRate);
    }

    treeBuilder_ = std::make_unique<LeafwiseTreeBuilder>(
        config_, createOptimalSplitFinder(), createCriterion());
}

void LightGBMTrainer::train(const std::vector<double>& data,
                            int rowLength,
                            const std::vector<double>& labels) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    size_t n = labels.size();

    if (config_.verbose) {
        std::cout << "LightGBM Enhanced: " << n << " samples, " << rowLength << " features" << std::endl;
        std::cout << "Split method: " << config_.splitMethod << std::endl;
        std::cout << "GOSS: " << (config_.enableGOSS ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Feature Bundling: " << (config_.enableFeatureBundling ? "Enabled" : "Disabled") << std::endl;
    }

    // **并行优化1: 特征绑定的并行处理**
    preprocessFeaturesParallel(data, rowLength, n);

    // 初始化
    double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    gradients_.resize(n);

    // **并行优化2: Boosting主循环优化**
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // **并行优化3: 损失计算的并行**
        double currentLoss = computeLossParallel(labels, predictions);
        trainingLoss_.push_back(currentLoss);

        // **并行优化4: 残差计算的并行**
        computeGradientsParallel(labels, predictions);

        // **并行优化5: GOSS采样的并行优化**
        if (config_.enableGOSS) {
            performGOSSSamplingParallel();
        } else {
            // 使用全量样本
            sampleIndices_.resize(n);
            sampleWeights_.assign(n, 1.0);
            std::iota(sampleIndices_.begin(), sampleIndices_.end(), 0);
        }

        // **并行优化6: 叶子构建的并行**
        auto tree = buildTreeParallel(data, rowLength, labels, gradients_, 
                                     sampleIndices_, sampleWeights_, featureBundles_);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": No valid split, stopping." << std::endl;
            }
            break;
        }

        // **并行优化7: 预测更新的并行**
        updatePredictionsParallel(data, rowLength, tree.get(), predictions);

        model_.addTree(std::move(tree), config_.learningRate);

        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);

        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }

        // **并行优化8: 早停检查的优化**
        if (config_.earlyStoppingRounds > 0 && shouldEarlyStopParallel(iter)) {
            if (config_.verbose) {
                std::cout << "Early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    if (config_.verbose) {
        std::cout << "LightGBM Enhanced training completed in " << totalTime.count() 
                  << "ms with " << model_.getTreeCount() << " trees" << std::endl;
        
        #ifdef _OPENMP
        std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(n * config_.numIterations) 
                      / (totalTime.count() * omp_get_max_threads())) 
                  << " samples/(ms*thread)" << std::endl;
        #endif
    }
}

// **新增方法：并行特征预处理**
void LightGBMTrainer::preprocessFeaturesParallel(const std::vector<double>& data,
                                                 int rowLength,
                                                 size_t sampleSize) {
    if (!config_.enableFeatureBundling) {
        // 简化版：每个特征自己成一个bundle
        featureBundles_.clear();
        featureBundles_.reserve(rowLength);
        
        // **并行优化9: 特征bundle创建的并行**
        std::vector<FeatureBundle> tempBundles(rowLength);
        
        #pragma omp parallel for schedule(static) if(rowLength > 8)
        for (int i = 0; i < rowLength; ++i) {
            tempBundles[i].features.push_back(i);
            tempBundles[i].offsets.push_back(0.0);
            tempBundles[i].totalBins = config_.maxBin;
        }
        
        featureBundles_ = std::move(tempBundles);
        
        if (config_.verbose) {
            std::cout << "Feature bundling: " << rowLength << " -> "
                      << featureBundles_.size() << " bundles (parallel)" << std::endl;
        }
    } else {
        // 使用特征绑定器
        featureBundler_->createBundles(data, rowLength, sampleSize, featureBundles_);
    }
}

// **新增方法：并行损失计算**
double LightGBMTrainer::computeLossParallel(const std::vector<double>& labels,
                                           const std::vector<double>& predictions) const {
    size_t n = labels.size();
    double totalLoss = 0.0;
    
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += lossFunction_->loss(labels[i], predictions[i]);
    }
    
    return totalLoss / n;
}

// **新增方法：并行梯度计算**
void LightGBMTrainer::computeGradientsParallel(const std::vector<double>& labels,
                                              const std::vector<double>& predictions) {
    size_t n = labels.size();
    
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        gradients_[i] = labels[i] - predictions[i];
    }
}

// **新增方法：并行GOSS采样**
void LightGBMTrainer::performGOSSSamplingParallel() {
    // 计算梯度绝对值
    size_t n = gradients_.size();
    std::vector<double> absGradients(n);
    
    #pragma omp parallel for schedule(static) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        absGradients[i] = std::abs(gradients_[i]);
    }
    
    // 执行GOSS采样
    gossSampler_->sample(absGradients, sampleIndices_, sampleWeights_);

    // **并行优化10: 权重归一化的并行**
    if (!sampleWeights_.empty()) {
        double totalWeight = 0.0;
        
        #pragma omp parallel for reduction(+:totalWeight) schedule(static)
        for (size_t i = 0; i < sampleWeights_.size(); ++i) {
            totalWeight += sampleWeights_[i];
        }
        
        if (totalWeight > 0) {
            double normFactor = static_cast<double>(n) / totalWeight;
            
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < sampleWeights_.size(); ++i) {
                sampleWeights_[i] *= normFactor;
            }
        }
    }
}

// **新增方法：并行树构建**
std::unique_ptr<Node> LightGBMTrainer::buildTreeParallel(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& labels,
    const std::vector<double>& targets,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights,
    const std::vector<FeatureBundle>& bundles) const {
    
    // 使用优化的叶子构建器
    return treeBuilder_->buildTree(data, rowLength, labels, targets,
                                  sampleIndices, sampleWeights, bundles);
}

// **新增方法：并行预测更新**
void LightGBMTrainer::updatePredictionsParallel(const std::vector<double>& data,
                                               int rowLength,
                                               const Node* tree,
                                               std::vector<double>& predictions) const {
    size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        double treePred = predictSingleTree(tree, sample, rowLength);
        predictions[i] += config_.learningRate * treePred;
    }
}

// **新增方法：并行早停检查**
bool LightGBMTrainer::shouldEarlyStopParallel(int currentIter) const {
    if (currentIter < config_.earlyStoppingRounds) return false;
    
    // 检查最近几轮的损失是否改善
    auto recentStart = trainingLoss_.end() - config_.earlyStoppingRounds;
    auto recentEnd = trainingLoss_.end() - 1;
    
    double bestRecentLoss = *std::min_element(recentStart, recentEnd);
    double currentLoss = trainingLoss_.back();
    
    return currentLoss >= bestRecentLoss - config_.tolerance;
}

void LightGBMTrainer::evaluate(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y,
                               double& mse,
                               double& mae) {
    auto predictions = model_.predictBatch(X, rowLength);
    size_t n = y.size();

    // **并行优化11: 评估指标计算的并行**
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

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    size_t n = y.size();
    double sum = 0.0;
    
    // **并行优化12: 基准分数计算的并行**
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

std::vector<double> LightGBMTrainer::calculateFeatureImportance(int numFeatures) const {
    return model_.getFeatureImportance(numFeatures);
}

// 分割器工厂方法（保持不变）
std::unique_ptr<ISplitFinder> LightGBMTrainer::createOptimalSplitFinder() const {
    const std::string& method = config_.splitMethod;

    if (method == "histogram_ew" || method.find("histogram_ew:") == 0) {
        int bins = config_.histogramBins;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEWFinder>(bins);
    } else if (method == "histogram_eq" || method.find("histogram_eq:") == 0) {
        int bins = config_.histogramBins;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEQFinder>(bins);
    } else if (method == "adaptive_ew" || method.find("adaptive_ew:") == 0) {
        std::string rule = config_.adaptiveRule;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            rule = method.substr(pos + 1);
        }
        return std::make_unique<AdaptiveEWFinder>(8, config_.maxAdaptiveBins, rule);
    } else if (method == "adaptive_eq") {
        return std::make_unique<AdaptiveEQFinder>(
            config_.minSamplesPerBin,
            config_.maxAdaptiveBins,
            config_.variabilityThreshold);
    } else if (method == "exhaustive") {
        return std::make_unique<ExhaustiveSplitFinder>();
    } else {
        return std::make_unique<HistogramEWFinder>(config_.histogramBins);
    }
}

std::unique_ptr<ISplitFinder> LightGBMTrainer::createHistogramFinder() const {
    return std::make_unique<HistogramEWFinder>(config_.histogramBins);
}

std::unique_ptr<ISplitCriterion> LightGBMTrainer::createCriterion() const {
    return std::make_unique<MSECriterion>();
}

double LightGBMTrainer::predictSingleTree(const Node* tree,
                                          const double* sample,
                                          int /* rowLength */) const {
    const Node* cur = tree;
    while (cur && !cur->isLeaf) {
        int featureIndex = cur->getFeatureIndex();
        double value = sample[featureIndex];
        cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}

double LightGBMTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

void LightGBMTrainer::preprocessFeatures(const std::vector<double>& data,
                                         int rowLength,
                                         size_t sampleSize) {
    // 调用并行版本
    preprocessFeaturesParallel(data, rowLength, sampleSize);
}