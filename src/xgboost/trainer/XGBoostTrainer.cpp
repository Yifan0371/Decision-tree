#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

XGBoostTrainer::XGBoostTrainer(const XGBoostConfig& config) : config_(config) {
    lossFunction_ = XGBoostLossFactory::create(config_.objective);
    xgbCriterion_ = std::make_unique<XGBoostCriterion>(config_.lambda);
    trainingLoss_.reserve(config_.numRounds);
}

void XGBoostTrainer::train(const std::vector<double>& data, int rowLength, const std::vector<double>& labels) {
    const size_t n = labels.size();
    
    // **核心优化1: 列存储预处理 - 减少缓存缺失**
    ColumnData columnData(rowLength, n);
    
    // 并行构建每个特征的排序索引
    #pragma omp parallel for schedule(dynamic) if(rowLength > 4)
    for (int f = 0; f < rowLength; ++f) {
        columnData.sortedIndices[f].resize(n);
        std::iota(columnData.sortedIndices[f].begin(), columnData.sortedIndices[f].end(), 0);
        std::sort(columnData.sortedIndices[f].begin(), columnData.sortedIndices[f].end(),
                  [&](int a, int b) { return data[a * rowLength + f] < data[b * rowLength + f]; });
    }
    
    // 拷贝数据到列存储结构
    columnData.values = data;

    // 初始化模型和预测
    const double baseScore = computeBaseScore(labels);
    model_.setGlobalBaseScore(baseScore);

    std::vector<double> predictions(n, baseScore);
    std::vector<double> gradients(n), hessians(n);
    std::vector<char> rootMask(n, 1);

    // **核心优化2: Boosting主循环**
    for (int round = 0; round < config_.numRounds; ++round) {
        // 计算当前损失
        const double currentLoss = lossFunction_->computeBatchLoss(labels, predictions);
        trainingLoss_.push_back(currentLoss);

        // **优化3: 并行计算梯度和Hessian**
        lossFunction_->computeGradientsHessians(labels, predictions, gradients, hessians);

        // 行采样 - XGBoost subsample功能
        if (config_.subsample < 1.0) {
            const size_t sampleSize = static_cast<size_t>(n * config_.subsample);
            thread_local std::mt19937 gen(std::random_device{}());
            thread_local std::vector<int> indices(n);
            
            if (indices.size() != n) {
                indices.resize(n);
                std::iota(indices.begin(), indices.end(), 0);
            }
            
            std::shuffle(indices.begin(), indices.end(), gen);
            
            std::fill(rootMask.begin(), rootMask.end(), 0);
            for (size_t i = 0; i < sampleSize; ++i) {
                rootMask[indices[i]] = 1;
            }
        } else {
            std::fill(rootMask.begin(), rootMask.end(), 1);
        }

        // 训练单棵树
        auto tree = trainSingleTree(columnData, gradients, hessians, rootMask);
        if (!tree) break;

        // **优化4: 并行更新预测**
        updatePredictions(data, rowLength, tree.get(), predictions);
        model_.addTree(std::move(tree), config_.eta);

        // 早停检查
        if (hasValidation_ && config_.earlyStoppingRounds > 0) {
            if (shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) break;
        }
    }
}

std::unique_ptr<Node> XGBoostTrainer::trainSingleTree(const ColumnData& columnData,
                                                     const std::vector<double>& gradients,
                                                     const std::vector<double>& hessians,
                                                     const std::vector<char>& rootMask) const {
    auto root = std::make_unique<Node>();
    buildXGBNode(root.get(), columnData, gradients, hessians, rootMask, 0);
    return root;
}

void XGBoostTrainer::buildXGBNode(Node* node, 
                                  const ColumnData& columnData,
                                  const std::vector<double>& gradients,
                                  const std::vector<double>& hessians,
                                  const std::vector<char>& nodeMask, 
                                  int depth) const {
    const size_t n = nodeMask.size();

    // **优化5: 并行计算节点统计**
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    node->samples = sampleCount;
    const double leafWeight = xgbCriterion_->computeLeafWeight(G_parent, H_parent);

    // 停止条件检查
    if (depth >= config_.maxDepth || sampleCount < 2 || H_parent < config_.minChildWeight) {
        node->makeLeaf(leafWeight);
        return;
    }

    // **核心优化6: 使用XGBoost专用分裂查找**
    auto [bestFeature, bestThreshold, bestGain] = findBestSplitXGB(columnData, gradients, hessians, nodeMask);

    if (bestFeature < 0 || bestGain <= config_.gamma) {
        node->makeLeaf(leafWeight);
        return;
    }

    // 执行分裂
    node->makeInternal(bestFeature, bestThreshold);

    // **优化7: 并行构建子节点掩码**
    std::vector<char> leftMask(n, 0), rightMask(n, 0);
    
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (!nodeMask[i]) continue;
        const double val = columnData.values[i * columnData.numFeatures + bestFeature];
        if (val <= bestThreshold) {
            leftMask[i] = 1;
        } else {
            rightMask[i] = 1;
        }
    }

    // 创建子节点
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();

    // **优化8: 浅层并行递归**
    if (depth <= 2 && sampleCount > 5000) {
        #pragma omp parallel sections
        {
            #pragma omp section
            buildXGBNode(node->leftChild.get(), columnData, gradients, hessians, leftMask, depth + 1);
            #pragma omp section
            buildXGBNode(node->rightChild.get(), columnData, gradients, hessians, rightMask, depth + 1);
        }
    } else {
        // 串行递归，避免过度并行的开销
        buildXGBNode(node->leftChild.get(), columnData, gradients, hessians, leftMask, depth + 1);
        buildXGBNode(node->rightChild.get(), columnData, gradients, hessians, rightMask, depth + 1);
    }
}

std::tuple<int, double, double> XGBoostTrainer::findBestSplitXGB(
    const ColumnData& columnData,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask) const {

    const size_t n = nodeMask.size();

    // 计算父节点统计
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    if (sampleCount < 2 || H_parent < config_.minChildWeight) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

    // **核心优化9: 并行特征扫描**
    #pragma omp parallel if(columnData.numFeatures > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        // 线程局部缓冲区
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < columnData.numFeatures; ++f) {
            // **优化10: 高效构建节点内排序索引**
            nodeSorted.clear();
            const std::vector<int>& featureIndices = columnData.sortedIndices[f];
            
            for (const int idx : featureIndices) {
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

            // **优化11: 单次遍历计算最佳分裂**
            double G_left = 0.0, H_left = 0.0;
            
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = columnData.values[idx * columnData.numFeatures + f];
                const double nextVal = columnData.values[nextIdx * columnData.numFeatures + f];

                // 跳过相同特征值
                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

                // 检查左右子节点的Hessian约束
                if (H_left < config_.minChildWeight || H_right < config_.minChildWeight) continue;

                // **核心: 计算XGBoost增益**
                const double gain = xgbCriterion_->computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, config_.gamma);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
        // **优化12: 线程间归约**
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}

void XGBoostTrainer::updatePredictions(const std::vector<double>& data, int rowLength,
                                      const Node* tree, std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        const Node* cur = tree;
        
        // 快速树遍历
        while (cur && !cur->isLeaf) {
            const double val = sample[cur->getFeatureIndex()];
            cur = (val <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        
        if (cur) {
            predictions[i] += config_.eta * cur->getPrediction();
        }
    }
}

void XGBoostTrainer::evaluate(const std::vector<double>& X, int rowLength,
                              const std::vector<double>& y, double& mse, double& mae) {
    const auto predictions = model_.predictBatch(X, rowLength);
    const size_t n = y.size();

    mse = 0.0; 
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n; 
    mae /= n;
}

double XGBoostTrainer::computeBaseScore(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    return sum / n;
}

bool XGBoostTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    const double bestLoss = *std::min_element(losses.end() - patience - 1, losses.end() - 1);
    return losses.back() >= bestLoss - config_.tolerance;
}

double XGBoostTrainer::computeValidationLoss() const {
    if (!hasValidation_) return 0.0;
    const auto predictions = model_.predictBatch(X_val_, valRowLength_);
    return lossFunction_->computeBatchLoss(y_val_, predictions);
}

double XGBoostTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}