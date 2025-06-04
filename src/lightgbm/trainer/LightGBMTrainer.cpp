// =============================================================================
// src/lightgbm/trainer/LightGBMTrainer.cpp
// 深度 OpenMP 并行优化版本（阈值提高、合并并行、减少锁竞争）
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
    size_t n = labels.size();
    if (config_.verbose) {
        std::cout << "LightGBM Enhanced: " << n << " 样本, " << rowLength << " 特征" << std::endl;
        std::cout << "Split 方法: " << config_.splitMethod << std::endl;
        std::cout << "GOSS: " << (config_.enableGOSS ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Feature Bundling: " << (config_.enableFeatureBundling ? "Enabled" : "Disabled") << std::endl;
    }

    // 特征预处理：只在大规模特征时并行，否则串行
    if (!config_.enableFeatureBundling && rowLength >= 1000) {
        // 并行构造简单 bundle
        featureBundles_.clear();
        featureBundles_.reserve(rowLength);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rowLength; ++i) {
            FeatureBundle bundle;
            bundle.features.push_back(i);
            bundle.offsets.push_back(0.0);
            bundle.totalBins = config_.maxBin;
            #pragma omp critical
            {
                featureBundles_.push_back(bundle);
            }
        }
        if (config_.verbose) {
            std::cout << "Feature Bundling (parallel): " << rowLength << " -> "
                      << featureBundles_.size() << " bundles" << std::endl;
        }
    } else {
        // 串行或者使用 Bundler
        preprocessFeaturesSerial(data, rowLength, n);
    }

    // 初始化预测和梯度
    double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    gradients_.assign(n, 0.0);

    // Boosting 迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // 串行计算损失并更新梯度（阈值 n < 10000 都走串行）
        double currentLoss = computeLossSerial(labels, predictions);
        trainingLoss_.push_back(currentLoss);
        computeGradientsSerial(labels, predictions);

        // GOSS 采样或全量
        if (config_.enableGOSS) {
            std::vector<double> absGradients(n);
            // 串行计算 absGradients（阈值 n < 10000）
            for (size_t i = 0; i < n; ++i) {
                absGradients[i] = std::abs(gradients_[i]);
            }
            gossSampler_->sample(absGradients, sampleIndices_, sampleWeights_);
            // 串行归一化
            double totalWeight = std::accumulate(sampleWeights_.begin(), sampleWeights_.end(), 0.0);
            if (totalWeight > 0.0) {
                double normFactor = static_cast<double>(n) / totalWeight;
                for (double& w : sampleWeights_) {
                    w *= normFactor;
                }
            }
        } else {
            sampleIndices_.resize(n);
            sampleWeights_.assign(n, 1.0);
            std::iota(sampleIndices_.begin(), sampleIndices_.end(), 0);
        }

        // 构建一棵树（内部已并行优化）
        auto tree = treeBuilder_->buildTree(
            data, rowLength, labels, gradients_,
            sampleIndices_, sampleWeights_, featureBundles_);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": 找不到有效 split，停止训练。" << std::endl;
            }
            break;
        }

        // 更新预测（阈值 n >= 10000 并行，否则串行）
        if (n >= 10000) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const double* sample = &data[i * rowLength];
                double treePred = predictSingleTree(tree.get(), sample, rowLength);
                predictions[i] += config_.learningRate * treePred;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                const double* sample = &data[i * rowLength];
                double treePred = predictSingleTree(tree.get(), sample, rowLength);
                predictions[i] += config_.learningRate * treePred;
            }
        }

        model_.addTree(std::move(tree), config_.learningRate);

        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);

        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << " ms" << std::endl;
        }

        // 早停：用串行滑动窗口最小值判断，避免每轮都 O(k) 扫描
        if (config_.earlyStoppingRounds > 0 && iter >= config_.earlyStoppingRounds) {
            double recentBest = *std::min_element(
                trainingLoss_.end() - config_.earlyStoppingRounds,
                trainingLoss_.end());
            if (currentLoss >= recentBest - config_.tolerance) {
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
    // 串行或并行评估，阈值 n >= 10000
    double sumMSE = 0.0, sumMAE = 0.0;
    if (n >= 10000) {
        #pragma omp parallel for reduction(+:sumMSE, sumMAE) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double diff = y[i] - predictions[i];
            sumMSE += diff * diff;
            sumMAE += std::abs(diff);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double diff = y[i] - predictions[i];
            sumMSE += diff * diff;
            sumMAE += std::abs(diff);
        }
    }
    mse = sumMSE / n;
    mae = sumMAE / n;
}

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

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    size_t n = y.size();
    double sum = 0.0;
    if (n >= 10000) {
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            sum += y[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            sum += y[i];
        }
    }
    return sum / n;
}

double LightGBMTrainer::computeLossSerial(const std::vector<double>& labels,
                                          const std::vector<double>& predictions) const {
    size_t n = labels.size();
    double loss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        loss += lossFunction_->loss(labels[i], predictions[i]);
    }
    return loss / n;
}

void LightGBMTrainer::computeGradientsSerial(const std::vector<double>& labels,
                                             const std::vector<double>& predictions) {
    size_t n = labels.size();
    for (size_t i = 0; i < n; ++i) {
        gradients_[i] = labels[i] - predictions[i];
    }
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
        int featureIndex = cur->getFeatureIndex();
        double value = sample[featureIndex];
        cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}
