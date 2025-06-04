// =============================================================================
// src/xgboost/trainer/XGBoostTrainer.cpp - 优化版本（避免vector<vector>和new）
// =============================================================================
#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// 注意：此处不再定义 OptimizedSortedIndices，
// 因为它已经在 XGBoostSplitFinder.hpp 中定义过了。

XGBoostTrainer::XGBoostTrainer(const XGBoostConfig& config)
    : config_(config) {
    // 创建损失函数
    lossFunction_ = XGBoostLossFactory::create(config_.objective);
    // 创建 XGBoost 特有组件
    xgbCriterion_ = std::make_unique<XGBoostCriterion>(config_.lambda);
    xgbFinder_ = std::make_unique<XGBoostSplitFinder>(config_.gamma, config_.minChildWeight);

    trainingLoss_.reserve(config_.numRounds);
    if (config_.earlyStoppingRounds > 0) {
        validationLoss_.reserve(config_.numRounds);
    }
    
    #ifdef _OPENMP
    if (config_.verbose) {
        std::cout << "XGBoost initialized with OpenMP support (" 
                  << omp_get_max_threads() << " threads)" << std::endl;
    }
    #endif
}

void XGBoostTrainer::train(const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    const size_t n = labels.size();

    if (config_.verbose) {
        std::cout << "Starting XGBoost training: " << n << " samples, " 
                  << rowLength << " features, " << config_.numRounds << " rounds" << std::endl;
    }

    // ======================================================
    // **优化1: 使用优化的索引存储结构替代vector<vector>**
    // ======================================================
    OptimizedSortedIndices sortedIndices(rowLength, n);
    
    #pragma omp parallel for schedule(dynamic) if(rowLength > 4)
    for (int f = 0; f < rowLength; ++f) {
        auto [start, end] = sortedIndices.getFeatureRange(f);
        std::iota(start, end, 0);
        std::sort(start, end, [&](int a, int b) {
            return data[a * rowLength + f] < data[b * rowLength + f];
        });
    }

    // 初始化根节点掩码
    std::vector<char> rootMask(n, 1);

    // 计算基准分数并初始化
    const double baseScore = computeBaseScore(labels);
    model_.setGlobalBaseScore(baseScore);

    std::vector<double> predictions(n, baseScore);
    std::vector<double> gradients(n), hessians(n);

    // ======================================================
    // **优化2: Boosting主循环优化**
    // ======================================================
    for (int round = 0; round < config_.numRounds; ++round) {
        auto roundStart = std::chrono::high_resolution_clock::now();

        // **优化3: 损失计算的并行**
        const double currentLoss = lossFunction_->computeBatchLoss(labels, predictions);
        trainingLoss_.push_back(currentLoss);

        // **优化4: 梯度和Hessian计算的并行**
        lossFunction_->computeGradientsHessians(labels, predictions, gradients, hessians);

        // 调试输出和性能监控
        if (config_.verbose && (round <= 2 || round % 20 == 0)) {
            double totalGrad = 0.0, totalHess = 0.0;
            
            #pragma omp parallel for reduction(+:totalGrad,totalHess) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                totalGrad += std::abs(gradients[i]);
                totalHess += hessians[i];
            }
            
            std::cout << "Round " << round 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | AvgGrad: " << (totalGrad / n)
                      << " | TotalHess: " << totalHess << std::endl;
        }

        // 行采样与列采样
        std::vector<int> sampleIndices, featureIndices;
        sampleData(data, rowLength, gradients, hessians, sampleIndices, featureIndices);

        // **优化5: 使用优化的树训练**
        auto tree = trainSingleTreeOptimized(data, rowLength, gradients, hessians,
                                            rootMask, sortedIndices);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Round " << round << ": 未找到有效分裂，停止训练。" << std::endl;
            }
            break;
        }

        // **优化6: 预测更新的并行**
        updatePredictionsParallel(data, rowLength, tree.get(), predictions);

        // 将新树加入模型
        model_.addTree(std::move(tree), config_.eta);

        auto roundEnd = std::chrono::high_resolution_clock::now();
        auto roundTime = std::chrono::duration_cast<std::chrono::milliseconds>(roundEnd - roundStart);
        
        if (config_.verbose && round % 10 == 0) {
            std::cout << "Round " << round
                      << " | Time: " << roundTime.count() << "ms"
                      << " | Trees: " << model_.getTreeCount() << std::endl;
        }

        // **优化7: 收敛检查**
        if (round > 10 && shouldConverge(gradients)) {
            if (config_.verbose) {
                std::cout << "Converged at round " << round << std::endl;
            }
            break;
        }

        // 验证集早停
        if (hasValidation_ && config_.earlyStoppingRounds > 0) {
            const double valLoss = computeValidationLoss();
            validationLoss_.push_back(valLoss);
            
            if (shouldEarlyStop(validationLoss_, config_.earlyStoppingRounds)) {
                if (config_.verbose) {
                    std::cout << "Early stopping at round " << round 
                              << " (val_loss: " << valLoss << ")" << std::endl;
                }
                break;
            }
        }
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    if (config_.verbose) {
        std::cout << "XGBoost training completed in " << totalTime.count() 
                  << "ms with " << model_.getTreeCount() << " trees" << std::endl;
        
        #ifdef _OPENMP
        std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(n * config_.numRounds) 
                      / (totalTime.count() * omp_get_max_threads())) 
                  << " samples/(ms*thread)" << std::endl;
        #endif
    }
}

// **新增方法：优化的树训练（避免vector<vector>）**
std::unique_ptr<Node> XGBoostTrainer::trainSingleTreeOptimized(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& rootMask,
    const OptimizedSortedIndices& sortedIndices) const {

    // 使用智能指针创建根节点
    auto root = std::make_unique<Node>();
    
    // 递归构建整棵树
    buildXGBNodeOptimized(root.get(),
                         X, rowLength, gradients, hessians,
                         rootMask, sortedIndices,
                         /*depth=*/0);
    return root;
}

// **新增方法：优化的节点构建（避免vector<vector>）**
void XGBoostTrainer::buildXGBNodeOptimized(Node* node,
                                          const std::vector<double>& X,
                                          int rowLength,
                                          const std::vector<double>& gradients,
                                          const std::vector<double>& hessians,
                                          const std::vector<char>& nodeMask,
                                          const OptimizedSortedIndices& sortedIndices,
                                          int depth) const {

    const size_t n = nodeMask.size();

    // **优化8: 节点统计计算的并行**
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    node->samples = sampleCount;
    node->metric = xgbCriterion_->computeStructureScore(G_parent, H_parent);

    // 计算叶节点权重
    const double leafWeight = xgbCriterion_->computeLeafWeight(G_parent, H_parent);

    // 停止条件检查
    if (depth >= config_.maxDepth || sampleCount < 2 || H_parent < config_.minChildWeight) {
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }

    // 寻找最优分裂（使用优化的方法）
    auto [bestFeature, bestThreshold, bestGain] = 
        findBestSplitOptimized(X, rowLength, gradients, hessians, nodeMask, sortedIndices);

    if (bestFeature < 0 || bestGain <= 0.0) {
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }

    // 执行分裂
    node->makeInternal(bestFeature, bestThreshold);

    // **优化9: 子节点掩码构建的并行**
    std::vector<char> leftMask(n, 0), rightMask(n, 0);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (!nodeMask[i]) continue;
        const double val = X[i * rowLength + bestFeature];
        if (val <= bestThreshold) {
            leftMask[i] = 1;
        } else {
            rightMask[i] = 1;
        }
    }

    // 使用智能指针创建子节点
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();

    // **优化10: 子树构建的并行**
    const bool useParallelSections = (depth <= 2) && (sampleCount > 5000);
    
    if (useParallelSections) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                buildXGBNodeOptimized(node->leftChild.get(),
                                    X, rowLength, gradients, hessians,
                                    leftMask, sortedIndices,
                                    depth + 1);
            }
            #pragma omp section
            {
                buildXGBNodeOptimized(node->rightChild.get(),
                                    X, rowLength, gradients, hessians,
                                    rightMask, sortedIndices,
                                    depth + 1);
            }
        }
    } else {
        // 串行递归
        buildXGBNodeOptimized(node->leftChild.get(),
                            X, rowLength, gradients, hessians,
                            leftMask, sortedIndices,
                            depth + 1);
        buildXGBNodeOptimized(node->rightChild.get(),
                            X, rowLength, gradients, hessians,
                            rightMask, sortedIndices,
                            depth + 1);
    }
}

// **新增方法：优化的分裂查找（避免vector<vector>）**
std::tuple<int, double, double> XGBoostTrainer::findBestSplitOptimized(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const OptimizedSortedIndices& sortedIndices) const {

    const size_t n = nodeMask.size();

    // 计算当前节点的统计信息
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

    // 遍历每个特征
    for (int f = 0; f < rowLength; ++f) {
        // 获取当前特征的排序索引
        const int* featureIndices = sortedIndices.getFeatureData(f);
        const size_t featureSize = sortedIndices.getFeatureSize();
        
        // 构造当前节点在特征f上的有序索引列表
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        for (size_t i = 0; i < featureSize; ++i) {
            const int idx = featureIndices[i];
            if (nodeMask[idx]) {
                nodeSorted.push_back(idx);
            }
        }
        
        if (nodeSorted.size() < 2) continue;

        // 遍历分裂点
        double G_left = 0.0, H_left = 0.0;
        for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
            const int idx = nodeSorted[i];
            G_left += gradients[idx];
            H_left += hessians[idx];

            const int nextIdx = nodeSorted[i + 1];
            const double currentVal = data[idx * rowLength + f];
            const double nextVal = data[nextIdx * rowLength + f];

            // 跳过相同特征值
            if (std::abs(nextVal - currentVal) < EPS) continue;

            const double G_right = G_parent - G_left;
            const double H_right = H_parent - H_left;

            // 检查约束
            if (H_left < config_.minChildWeight || H_right < config_.minChildWeight) continue;

            // 计算增益
            const double gain = xgbCriterion_->computeSplitGain(
                G_left, H_left, G_right, H_right, G_parent, H_parent, config_.gamma);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = f;
                bestThreshold = 0.5 * (currentVal + nextVal);
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}

// **优化的预测更新方法**
void XGBoostTrainer::updatePredictionsParallel(const std::vector<double>& data,
                                               int rowLength,
                                               const Node* tree,
                                               std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        double treePred = 0.0;
        
        // 单棵树预测
        const Node* cur = tree;
        while (cur && !cur->isLeaf) {
            const double val = sample[cur->getFeatureIndex()];
            cur = (val <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        if (cur) {
            treePred = cur->getPrediction();
        }
        
        predictions[i] += config_.eta * treePred;
    }
}

bool XGBoostTrainer::shouldConverge(const std::vector<double>& gradients) const {
    const size_t n = gradients.size();
    double totalGrad = 0.0;
    
    #pragma omp parallel for reduction(+:totalGrad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        totalGrad += std::abs(gradients[i]);
    }
    
    return (totalGrad / n) < 1e-8;
}

void XGBoostTrainer::evaluate(const std::vector<double>& X,
                              int rowLength,
                              const std::vector<double>& y,
                              double& mse,
                              double& mae) {
    const auto predictions = model_.predictBatch(X, rowLength);
    const size_t n = y.size();

    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 1024) if(n > 2000)
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
    
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

void XGBoostTrainer::sampleData(const std::vector<double>& /* X */,
                                int /* rowLength */,
                                const std::vector<double>& gradients,
                                const std::vector<double>& /* hessians */,
                                std::vector<int>& sampleIndices,
                                std::vector<int>& /* featureIndices */) const {

    sampleIndices.clear();
    if (config_.subsample >= 1.0) {
        return;
    }
    
    const size_t n = gradients.size();
    const size_t sampleSize = static_cast<size_t>(n * config_.subsample);

    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::vector<int> allIndices;
    
    if (allIndices.size() != n) {
        allIndices.resize(n);
        std::iota(allIndices.begin(), allIndices.end(), 0);
    }
    
    std::shuffle(allIndices.begin(), allIndices.end(), gen);
    sampleIndices.assign(allIndices.begin(), allIndices.begin() + sampleSize);
}

bool XGBoostTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    const double bestLoss = *std::min_element(losses.end() - patience - 1, losses.end() - 1);
    const double currentLoss = losses.back();
    return currentLoss >= bestLoss - config_.tolerance;
}

double XGBoostTrainer::computeValidationLoss() const {
    if (!hasValidation_) return 0.0;
    const auto predictions = model_.predictBatch(X_val_, valRowLength_);
    return lossFunction_->computeBatchLoss(y_val_, predictions);
}

double XGBoostTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

// 保留旧接口的兼容性
std::unique_ptr<Node> XGBoostTrainer::trainSingleTree(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& rootMask,
    const std::vector<std::vector<int>>& sortedIndicesAll) const {
    
    // 转换为优化的数据结构
    OptimizedSortedIndices optimizedIndices(rowLength, gradients.size());
    for (int f = 0; f < rowLength; ++f) {
        auto [start, end] = optimizedIndices.getFeatureRange(f);
        std::copy(sortedIndicesAll[f].begin(), sortedIndicesAll[f].end(), start);
    }
    
    return trainSingleTreeOptimized(X, rowLength, gradients, hessians, rootMask, optimizedIndices);
}
