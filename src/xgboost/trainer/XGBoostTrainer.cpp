#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

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
}

void XGBoostTrainer::train(const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels) {
    size_t n = labels.size();

    // ======================================================
    // 1. 全局预排序：针对每个特征 f，对所有样本进行一次升序排序
    //    sortedIndicesAll[f][i] 表示：特征 f 上第 i 小的样本全局索引
    // ======================================================
    std::vector<std::vector<int>> sortedIndicesAll(rowLength);
    for (int f = 0; f < rowLength; ++f) {
        sortedIndicesAll[f].resize(n);
        std::iota(sortedIndicesAll[f].begin(), sortedIndicesAll[f].end(), 0);
        std::sort(sortedIndicesAll[f].begin(), sortedIndicesAll[f].end(),
                  [&](int a, int b) {
                      return data[a * rowLength + f] < data[b * rowLength + f];
                  });
    }

    // ======================================================
    // 2. 初始化根节点掩码 rootMask：长度 = n，所有样本都在根节点（标记为 1）
    // ======================================================
    std::vector<char> rootMask(n, 1);

    // ======================================================
    // 3. 计算基准分数并初始化模型、预测值、梯度、Hessian
    // ======================================================
    double baseScore = computeBaseScore(labels);
    model_.setGlobalBaseScore(baseScore);

    std::vector<double> predictions(n, baseScore);
    std::vector<double> gradients(n), hessians(n);

    // ======================================================
    // 4. Boosting 迭代
    // ======================================================
    for (int round = 0; round < config_.numRounds; ++round) {
        auto roundStart = std::chrono::high_resolution_clock::now();

        // 4.1 计算当前训练集损失
        double currentLoss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            currentLoss += lossFunction_->loss(labels[i], predictions[i]);
        }
        currentLoss /= n;
        trainingLoss_.push_back(currentLoss);

        // 4.2 计算梯度和 Hessian（向量化易由编译器处理）
        lossFunction_->computeGradientsHessians(labels, predictions, gradients, hessians);

        // 4.3 调试输出（仅前几轮或间隔输出）
        if (config_.verbose) {
            if (round <= 2 || round % 20 == 0) {
                double totalGrad = 0.0, totalHess = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    totalGrad += std::abs(gradients[i]);
                    totalHess += hessians[i];
                }
                std::cout << "DEBUG: 第 " << round << " 轮 | 总梯度=" 
                          << std::fixed << std::setprecision(6) << totalGrad
                          << " | 平均梯度=" << (totalGrad / n)
                          << " | 总Hessian=" << totalHess << std::endl;
                if (round <= 2) {
                    std::cout << "DEBUG: 前3个样本 label/pred/grad/hess:" << std::endl;
                    for (int i = 0; i < std::min(3, static_cast<int>(n)); ++i) {
                        std::cout << "  样本" << i << ": "
                                  << labels[i] << " / "
                                  << predictions[i] << " / "
                                  << gradients[i] << " / "
                                  << hessians[i] << std::endl;
                    }
                }
            }
        }

        // 4.4 行采样与列采样（保持原逻辑，不使用此处示例）
        std::vector<int> sampleIndices, featureIndices;
        sampleData(data, rowLength, gradients, hessians, sampleIndices, featureIndices);

        // 4.5 训练新树：传入根节点掩码和全局预排序结果
        auto tree = trainSingleTree(data, rowLength, gradients, hessians,
                                    rootMask, sortedIndicesAll);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Round " << round << ": 未找到有效分裂，停止训练。" << std::endl;
            }
            break;
        }

        // 4.6 更新预测值：遍历每个样本，用新树的输出更新 predictions
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &data[i * rowLength];
            double treePred = 0.0;
            // 单棵树预测
            const Node* cur = tree.get();
            while (cur && !cur->isLeaf) {
                double val = sample[cur->getFeatureIndex()];
                cur = (val <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
            }
            if (cur) {
                treePred = cur->getPrediction();
            }
            predictions[i] += config_.eta * treePred;
        }

        // 4.7 将新树加入模型
        model_.addTree(std::move(tree), config_.eta);

        auto roundEnd = std::chrono::high_resolution_clock::now();
        auto roundTime = std::chrono::duration_cast<std::chrono::milliseconds>(roundEnd - roundStart);
        if (config_.verbose && round % 10 == 0) {
            std::cout << "Round " << round
                      << " | Train Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Time: " << roundTime.count() << "ms" << std::endl;
        }

        // 4.8 收敛检查（梯度范数）
        if (round > 10) {
            double totalGrad = 0.0;
            for (double g : gradients) totalGrad += std::abs(g);
            if (totalGrad / n < 1e-8) {
                if (config_.verbose) {
                    std::cout << "Converged at round " << round
                              << " (gradient norm: " << (totalGrad / n) << ")" << std::endl;
                }
                break;
            }
        }
    }

    if (config_.verbose) {
        std::cout << "XGBoost training completed: "
                  << model_.getTreeCount() << " 树" << std::endl;
    }
}

std::unique_ptr<Node> XGBoostTrainer::trainSingleTree(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& rootMask,
    const std::vector<std::vector<int>>& sortedIndicesAll) const {

    // 创建根节点
    auto root = std::make_unique<Node>();
    // 递归构建整棵树
    buildXGBNode(root.get(),
                 X, rowLength, gradients, hessians,
                 rootMask, sortedIndicesAll,
                 /*depth=*/0);
    return root;
}

void XGBoostTrainer::buildXGBNode(Node* node,
                                  const std::vector<double>& X,
                                  int rowLength,
                                  const std::vector<double>& gradients,
                                  const std::vector<double>& hessians,
                                  const std::vector<char>& nodeMask,
                                  const std::vector<std::vector<int>>& sortedIndicesAll,
                                  int depth) const {

    size_t n = nodeMask.size();

    // 1. 计算当前节点的 G_parent, H_parent 及样本数
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
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
    double leafWeight = xgbCriterion_->computeLeafWeight(G_parent, H_parent);

    // 2. 停止条件：深度、样本数、Hessian 约束
    if (depth >= config_.maxDepth || sampleCount < 2 || H_parent < config_.minChildWeight) {
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }

    // 3. 在当前节点上寻找最优分裂
    auto [bestFeature, bestThreshold, bestGain] =
        xgbFinder_->findBestSplitXGB(
            X, rowLength, gradients, hessians,
            nodeMask, sortedIndicesAll,
            *xgbCriterion_);

    if (bestFeature < 0 || bestGain <= 0.0) {
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }

    // 4. 执行分裂：设置节点为内部节点
    node->makeInternal(bestFeature, bestThreshold);

    // 新建左右子节点掩码
    std::vector<char> leftMask(n, 0), rightMask(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (!nodeMask[i]) continue;
        double val = X[i * rowLength + bestFeature];
        if (val <= bestThreshold) {
            leftMask[i] = 1;
        } else {
            rightMask[i] = 1;
        }
    }

    // 5. 递归构建左右子树
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();

    buildXGBNode(node->leftChild.get(),
                 X, rowLength, gradients, hessians,
                 leftMask, sortedIndicesAll,
                 depth + 1);

    buildXGBNode(node->rightChild.get(),
                 X, rowLength, gradients, hessians,
                 rightMask, sortedIndicesAll,
                 depth + 1);
}

double XGBoostTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

void XGBoostTrainer::evaluate(const std::vector<double>& X,
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

double XGBoostTrainer::computeBaseScore(const std::vector<double>& y) const {
    return std::accumulate(y.begin(), y.end(), 0.0) / y.size();
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
    size_t n = gradients.size();
    size_t sampleSize = static_cast<size_t>(n * config_.subsample);

    // 原地构造全局索引并 shuffle，然后取前 sampleSize 个
    static std::mt19937 gen(std::random_device{}());
    static std::vector<int> allIndices;
    if (allIndices.size() != n) {
        allIndices.resize(n);
        std::iota(allIndices.begin(), allIndices.end(), 0);
    }
    std::shuffle(allIndices.begin(), allIndices.end(), gen);
    sampleIndices.assign(allIndices.begin(), allIndices.begin() + sampleSize);
}

bool XGBoostTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    double bestLoss = *std::min_element(losses.end() - patience - 1, losses.end() - 1);
    double currentLoss = losses.back();
    return currentLoss >= bestLoss - config_.tolerance;
}

double XGBoostTrainer::computeValidationLoss() const {
    if (!hasValidation_) return 0.0;
    auto predictions = model_.predictBatch(X_val_, valRowLength_);
    double totalLoss = 0.0;
    for (size_t i = 0; i < y_val_.size(); ++i) {
        totalLoss += lossFunction_->loss(y_val_[i], predictions[i]);
    }
    return totalLoss / y_val_.size();
}
