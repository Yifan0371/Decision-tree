// =============================================================================
// src/lightgbm/trainer/LightGBMTrainer.cpp - 优化版本（避免vector<vector>）
// =============================================================================
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "criterion/MSECriterion.hpp"
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

// 优化的特征绑定存储结构 - 替代vector<FeatureBundle>
struct OptimizedFeatureBundles {
    std::vector<int> featureToBundle;     // 特征到bundle的映射
    std::vector<double> featureOffsets;   // 特征的偏移值
    std::vector<int> bundleSizes;         // 每个bundle的大小
    int numBundles = 0;
    
    OptimizedFeatureBundles(int numFeatures) 
        : featureToBundle(numFeatures), featureOffsets(numFeatures, 0.0) {
        // 简单初始化：每个特征单独一个bundle
        for (int i = 0; i < numFeatures; ++i) {
            featureToBundle[i] = i;
        }
        bundleSizes.resize(numFeatures, 1);
        numBundles = numFeatures;
    }
    
    std::pair<int, double> transformFeature(int originalFeature, double value) const {
        return {featureToBundle[originalFeature], value + featureOffsets[originalFeature]};
    }
};

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
        std::cout << "LightGBM 初始化，OpenMP 线程数: "
                  << omp_get_max_threads() << std::endl;
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
    const size_t n = labels.size();
    if (config_.verbose) {
        std::cout << "LightGBM Enhanced: " << n << " 样本, " << rowLength << " 特征" << std::endl;
        std::cout << "Split 方法: " << config_.splitMethod << std::endl;
        std::cout << "GOSS: " << (config_.enableGOSS ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Feature Bundling: " << (config_.enableFeatureBundling ? "Enabled" : "Disabled") << std::endl;
    }

    // **优化1: 使用优化的特征绑定结构**
    OptimizedFeatureBundles optimizedBundles(rowLength);
    
    if (config_.enableFeatureBundling && rowLength >= 100) {
        preprocessFeaturesOptimized(data, rowLength, n, optimizedBundles);
    } else {
        // 简单处理：每个特征独立
        if (config_.verbose) {
            std::cout << "Feature Bundling (simple): " << rowLength << " -> "
                      << rowLength << " bundles" << std::endl;
        }
    }

    // 初始化预测和梯度
    const double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    gradients_.assign(n, 0.0);

    // Boosting 迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // 计算损失并更新梯度
        const double currentLoss = computeLossOptimized(labels, predictions);
        trainingLoss_.push_back(currentLoss);
        computeGradientsOptimized(labels, predictions);

        // GOSS 采样或全量
        if (config_.enableGOSS) {
            std::vector<double> absGradients(n);
            computeAbsGradients(absGradients);
            gossSampler_->sample(absGradients, sampleIndices_, sampleWeights_);
            normalizeWeights(n);
        } else {
            prepareFullSample(n);
        }

        // 构建一棵树
        auto tree = treeBuilder_->buildTree(
            data, rowLength, labels, gradients_,
            sampleIndices_, sampleWeights_, featureBundles_);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": 找不到有效 split，停止训练。" << std::endl;
            }
            break;
        }

        // **优化2: 高效预测更新**
        updatePredictionsOptimized(data, rowLength, tree.get(), predictions, n);
        model_.addTree(std::move(tree), config_.learningRate);

        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);

        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << " ms" << std::endl;
        }

        // 早停检查
        if (config_.earlyStoppingRounds > 0 && iter >= config_.earlyStoppingRounds) {
            if (checkEarlyStop(iter)) {
                if (config_.verbose) {
                    std::cout << "Early stopping at iteration " << iter << std::endl;
                }
                break;
            }
        }
    }

    if (config_.verbose) {
        std::cout << "LightGBM Enhanced 训练完成，共 " << model_.getTreeCount() << " 棵树" << std::endl;
    }
}

// **优化方法实现**

void LightGBMTrainer::preprocessFeaturesOptimized(const std::vector<double>& data,
                                                  int rowLength,
                                                  size_t sampleSize,
                                                  OptimizedFeatureBundles& bundles) {
    // 简化的特征绑定：基于稀疏性分析
    std::vector<double> sparsity(rowLength);
    constexpr double EPS = 1e-12;
    
    // **并行计算特征稀疏性**
    #pragma omp parallel for schedule(static) if(rowLength > 50)
    for (int f = 0; f < rowLength; ++f) {
        int nonZeroCount = 0;
        const size_t checkSize = std::min(sampleSize, size_t(10000)); // 采样检查
        
        for (size_t i = 0; i < checkSize; ++i) {
            if (std::abs(data[i * rowLength + f]) > EPS) {
                ++nonZeroCount;
            }
        }
        sparsity[f] = 1.0 - static_cast<double>(nonZeroCount) / checkSize;
    }
    
    // 根据稀疏性重新组织bundle
    constexpr double SPARSITY_THRESHOLD = 0.8;
    int bundleId = 0;
    
    for (int f = 0; f < rowLength; ++f) {
        if (sparsity[f] > SPARSITY_THRESHOLD) {
            // 高稀疏特征可以尝试绑定
            bundles.featureToBundle[f] = bundleId;
            bundles.featureOffsets[f] = 0.0; // 简化处理
        } else {
            // 低稀疏特征独立
            bundles.featureToBundle[f] = bundleId;
            bundles.featureOffsets[f] = 0.0;
        }
        ++bundleId;
    }
    
    bundles.numBundles = bundleId;
    bundles.bundleSizes.resize(bundleId, 1);
    
    if (config_.verbose) {
        std::cout << "Feature Bundling (optimized): " << rowLength << " -> "
                  << bundles.numBundles << " bundles" << std::endl;
    }
}

double LightGBMTrainer::computeLossOptimized(const std::vector<double>& labels,
                                            const std::vector<double>& predictions) const {
    const size_t n = labels.size();
    double loss = 0.0;
    
    #pragma omp parallel for reduction(+:loss) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        loss += lossFunction_->loss(labels[i], predictions[i]);
    }
    
    return loss / n;
}

void LightGBMTrainer::computeGradientsOptimized(const std::vector<double>& labels,
                                               const std::vector<double>& predictions) {
    const size_t n = labels.size();
    
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        gradients_[i] = labels[i] - predictions[i];
    }
}

void LightGBMTrainer::computeAbsGradients(std::vector<double>& absGradients) const {
    const size_t n = gradients_.size();
    
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        absGradients[i] = std::abs(gradients_[i]);
    }
}

void LightGBMTrainer::normalizeWeights(size_t n) {
    const double totalWeight = std::accumulate(sampleWeights_.begin(), sampleWeights_.end(), 0.0);
    if (totalWeight > 0.0) {
        const double normFactor = static_cast<double>(n) / totalWeight;
        
        #pragma omp parallel for schedule(static) if(sampleWeights_.size() > 1000)
        for (size_t i = 0; i < sampleWeights_.size(); ++i) {
            sampleWeights_[i] *= normFactor;
        }
    }
}

void LightGBMTrainer::prepareFullSample(size_t n) {
    sampleIndices_.resize(n);
    sampleWeights_.assign(n, 1.0);
    std::iota(sampleIndices_.begin(), sampleIndices_.end(), 0);
}

void LightGBMTrainer::updatePredictionsOptimized(const std::vector<double>& data,
                                                 int rowLength,
                                                 const Node* tree,
                                                 std::vector<double>& predictions,
                                                 size_t n) const {
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        const double treePred = predictSingleTree(tree, sample, rowLength);
        predictions[i] += config_.learningRate * treePred;
    }
}

bool LightGBMTrainer::checkEarlyStop(int currentIter) const {
    const int patience = config_.earlyStoppingRounds;
    if (static_cast<int>(trainingLoss_.size()) < patience + 1) {
        return false;
    }
    
    const auto recentStart = trainingLoss_.end() - patience - 1;
    const auto recentEnd = trainingLoss_.end() - 1;
    const double bestLoss = *std::min_element(recentStart, recentEnd);
    const double currentLoss = trainingLoss_.back();
    
    return currentLoss >= bestLoss - config_.tolerance;
}

// **保留其他必要方法**

double LightGBMTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

void LightGBMTrainer::evaluate(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y,
                               double& mse,
                               double& mae) {
    const auto predictions = model_.predictBatch(X, rowLength);
    const size_t n = y.size();
    
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        const double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

std::vector<double> LightGBMTrainer::calculateFeatureImportance(int numFeatures) const {
    return model_.getFeatureImportance(numFeatures);
}

std::unique_ptr<ISplitCriterion> LightGBMTrainer::createCriterion() const {
    return std::make_unique<MSECriterion>();
}

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

double LightGBMTrainer::predictSingleTree(const Node* tree,
                                          const double* sample,
                                          int /* rowLength */) const {
    const Node* cur = tree;
    while (cur && !cur->isLeaf) {
        const int featureIndex = cur->getFeatureIndex();
        const double value = sample[featureIndex];
        cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}

// **兼容性方法（保留旧接口）**
void LightGBMTrainer::preprocessFeaturesSerial(const std::vector<double>& /* data */,
                                               int rowLength,
                                               size_t /* sampleSize */) {
    // 串行简单 bundle（每个特征各占一个 bundle）
    featureBundles_.clear();
    featureBundles_.reserve(rowLength);
    for (int i = 0; i < rowLength; ++i) {
        FeatureBundle bundle;
        bundle.features.push_back(i);
        bundle.offsets.push_back(0.0);
        bundle.totalBins = config_.maxBin;
        featureBundles_.push_back(std::move(bundle));
    }
    if (config_.verbose) {
        std::cout << "Feature Bundling (serial): " << rowLength << " -> "
                  << featureBundles_.size() << " bundles" << std::endl;
    }
}