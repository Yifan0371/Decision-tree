// LightGBMTrainer.cpp
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "criterion/MSECriterion.hpp"

// **新增：分割器include**
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

LightGBMTrainer::LightGBMTrainer(const LightGBMConfig& config)
    : config_(config) {
    initializeComponents();

    // 预分配内存池
    gradients_.reserve(50000);
    sampleIndices_.reserve(50000);
    sampleWeights_.reserve(50000);
    trainingLoss_.reserve(config_.numIterations);
}

void LightGBMTrainer::initializeComponents() {
    lossFunction_ = std::make_unique<SquaredLoss>();

    if (config_.enableGOSS) {
        gossSampler_ = std::make_unique<GOSSSampler>(config_.topRate, config_.otherRate);
    }

    if (config_.enableFeatureBundling) {
        featureBundler_ = std::make_unique<FeatureBundler>(config_.maxBin, config_.maxConflictRate);
    }

    // 使用分割器工厂创建 split finder 和 criterion
    treeBuilder_ = std::make_unique<LeafwiseTreeBuilder>(
        config_, createOptimalSplitFinder(), createCriterion());
}

void LightGBMTrainer::train(const std::vector<double>& data,
                            int rowLength,
                            const std::vector<double>& labels) {
    size_t n = labels.size();

    if (config_.verbose) {
        std::cout << "LightGBM Enhanced: " << n << " samples, " << rowLength << " features" << std::endl;
        std::cout << "Split method: " << config_.splitMethod << std::endl;
    }

    // 构造 featureBundles（简化版，每个特征自己成一个 bundle）
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
                  << featureBundles_.size() << " bundles" << std::endl;
    }

    // 初始化
    double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    gradients_.resize(n);

    // Boosting迭代
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // 计算当前迭代损失
        double currentLoss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            currentLoss += lossFunction_->loss(labels[i], predictions[i]);
        }
        currentLoss /= n;
        trainingLoss_.push_back(currentLoss);

        // 计算残差（梯度）
        for (size_t i = 0; i < n; ++i) {
            gradients_[i] = labels[i] - predictions[i];
        }

        // GOSS采样或全量样本
        if (config_.enableGOSS) {
            std::vector<double> absGradients(n);
            for (size_t i = 0; i < n; ++i) {
                absGradients[i] = std::abs(gradients_[i]);
            }
            gossSampler_->sample(absGradients, sampleIndices_, sampleWeights_);

            // 权重归一化
            double totalWeight = 0.0;
            for (double w : sampleWeights_) totalWeight += w;
            if (totalWeight > 0) {
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

        // 构建一棵树
        auto tree = treeBuilder_->buildTree(
            data, rowLength, labels, gradients_,
            sampleIndices_, sampleWeights_, featureBundles_);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": No valid split, stopping." << std::endl;
            }
            break;
        }

        // 更新预测值
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &data[i * rowLength];
            double treePred = predictSingleTree(tree.get(), sample, rowLength);
            predictions[i] += config_.learningRate * treePred;
        }

        model_.addTree(std::move(tree), config_.learningRate);

        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);

        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }

        // 早停检查
        if (config_.earlyStoppingRounds > 0 && iter >= config_.earlyStoppingRounds) {
            bool shouldStop = true;
            double recentBest = *std::min_element(
                trainingLoss_.end() - config_.earlyStoppingRounds,
                trainingLoss_.end());

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
        std::cout << "LightGBM Enhanced training completed: " << model_.getTreeCount()
                  << " trees" << std::endl;
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

void LightGBMTrainer::preprocessFeatures(const std::vector<double>& /* data */,
                                         int /* rowLength */,
                                         size_t /* sampleSize */) {
    // 暂时空实现，未来可根据需求补充特征预处理逻辑
}

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

std::vector<double> LightGBMTrainer::calculateFeatureImportance(int numFeatures) const {
    return model_.getFeatureImportance(numFeatures);
}

// **新增：分割器工厂方法实现**
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
        // 默认使用 HistogramEWFinder
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
