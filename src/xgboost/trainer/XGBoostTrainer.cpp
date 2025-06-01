
// =============================================================================
// src/xgboost/trainer/XGBoostTrainer.cpp
// =============================================================================
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
    
    // 创建XGBoost特有组件
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
    
    if (config_.verbose) {
        std::cout << "Training XGBoost with " << config_.numRounds 
                  << " rounds..." << std::endl;
        std::cout << "Objective: " << config_.objective 
                  << " | Learning rate: " << config_.eta << std::endl;
    }
    
    size_t n = labels.size();
    
    // 计算基准分数并初始化模型
    double baseScore = computeBaseScore(labels);
    model_.setGlobalBaseScore(baseScore);
    
    if (config_.verbose) {
        std::cout << "DEBUG: 样本数=" << n << " 特征数=" << (rowLength-1) 
                  << " 基准分数=" << std::fixed << std::setprecision(6) << baseScore << std::endl;
    }
    
    // 初始化预测值
    std::vector<double> predictions(n, baseScore);
    std::vector<double> gradients(n), hessians(n);
    
    // Boosting迭代
    for (int round = 0; round < config_.numRounds; ++round) {
        auto roundStart = std::chrono::high_resolution_clock::now();
        
        // 计算当前损失
        double currentLoss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            currentLoss += lossFunction_->loss(labels[i], predictions[i]);
        }
        currentLoss /= n;
        trainingLoss_.push_back(currentLoss);
        
        // 计算梯度和Hessian
        lossFunction_->computeGradientsHessians(labels, predictions, gradients, hessians);
        
        // 检查梯度统计
        if (round <= 2 || round % 20 == 0) {
            double totalGrad = 0.0, totalHess = 0.0;
            for (size_t i = 0; i < n; ++i) {
                totalGrad += std::abs(gradients[i]);
                totalHess += hessians[i];
            }
            
            if (config_.verbose) {
                std::cout << "DEBUG: 第" << round << "轮 总梯度=" 
                          << std::fixed << std::setprecision(6) << totalGrad 
                          << " 平均梯度=" << (totalGrad / n)
                          << " 总Hessian=" << totalHess << std::endl;
                
                if (round <= 2) {
                    std::cout << "DEBUG: 前3个样本 label/pred/grad/hess:" << std::endl;
                    for (int i = 0; i < std::min(3, static_cast<int>(n)); ++i) {
                        std::cout << "  样本" << i << ": " << labels[i] << "/" 
                                  << predictions[i] << "/" << gradients[i] << "/" << hessians[i] << std::endl;
                    }
                }
            }
        }
        
        // 数据采样（支持行采样和列采样）
        std::vector<int> sampleIndices, featureIndices;
        sampleData(data, rowLength, gradients, hessians, sampleIndices, featureIndices);
        
        // 训练新树
        auto tree = trainSingleTree(data, rowLength, gradients, hessians, sampleIndices);
        
        if (!tree) {
            if (config_.verbose) {
                std::cout << "Round " << round << ": No valid split found, stopping." << std::endl;
            }
            break;
        }
        
        // 更新预测值
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &data[i * rowLength];
            double treePred = 0.0;
            
            // 遍历树获取预测值
            const Node* cur = tree.get();
            while (cur && !cur->isLeaf) {
                double value = sample[cur->getFeatureIndex()];
                cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
            }
            if (cur) {
                treePred = cur->getPrediction();
            }
            
            predictions[i] += config_.eta * treePred;
        }
        
        // 添加树到模型
        model_.addTree(std::move(tree), config_.eta);
        
        auto roundEnd = std::chrono::high_resolution_clock::now();
        auto roundTime = std::chrono::duration_cast<std::chrono::milliseconds>(roundEnd - roundStart);
        
        // 输出训练信息
        if (config_.verbose && round % 10 == 0) {
            std::cout << "Round " << round << " | Train Loss: " 
                      << std::fixed << std::setprecision(6) << currentLoss
                      << " | Time: " << roundTime.count() << "ms" << std::endl;
        }
        
        // 收敛检查：如果梯度很小且损失不再下降，可以提前停止
        if (round > 10) {
            double totalGrad = 0.0;
            for (double g : gradients) totalGrad += std::abs(g);
            if (totalGrad / n < 1e-8) {
                if (config_.verbose) {
                    std::cout << "Converged at round " << round << " (gradient norm: " 
                              << (totalGrad / n) << ")" << std::endl;
                }
                break;
            }
        }
    }
    
    if (config_.verbose) {
        std::cout << "XGBoost training completed: " << model_.getTreeCount() 
                  << " trees" << std::endl;
    }
}


std::unique_ptr<Node> XGBoostTrainer::trainSingleTree(
    const std::vector<double>& X,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& sampleIndices) const {
    
    auto root = std::make_unique<Node>();
    
    // 使用全部样本或采样样本
    std::vector<int> indices;
    if (sampleIndices.empty()) {
        indices.resize(gradients.size());
        std::iota(indices.begin(), indices.end(), 0);
    } else {
        indices = sampleIndices;
    }
    
    buildXGBNode(root.get(), X, rowLength, gradients, hessians, indices, 0);
    return root;
}


void XGBoostTrainer::buildXGBNode(Node* node,
                                  const std::vector<double>& X,
                                  int rowLength,
                                  const std::vector<double>& gradients,
                                  const std::vector<double>& hessians,
                                  const std::vector<int>& indices,
                                  int depth) const {
    
    if (indices.empty()) {
        node->makeLeaf(0.0);
        return;
    }
    
    // 计算节点统计量
    double G = 0.0, H = 0.0;
    for (int idx : indices) {
        G += gradients[idx];
        H += hessians[idx];
    }
    
    node->samples = indices.size();
    node->metric = xgbCriterion_->computeStructureScore(G, H);
    
    // 计算最优叶节点权重
    double leafWeight = xgbCriterion_->computeLeafWeight(G, H);
    
    // 添加调试输出
    static int nodeCount = 0;
    nodeCount++;
    if (nodeCount <= 10) {
        std::cout << "DEBUG: 节点深度=" << depth << " 样本数=" << indices.size() 
                  << " G=" << std::fixed << std::setprecision(6) << G 
                  << " H=" << H << " 叶权重=" << leafWeight << std::endl;
    }
    
    // 停止条件检查
    if (depth >= config_.maxDepth ||
        indices.size() < 2 ||
        H < config_.minChildWeight) {
        
        if (nodeCount <= 10) {
            std::cout << "DEBUG: 停止分裂 - 深度=" << depth 
                      << "/" << config_.maxDepth << " 样本=" << indices.size() 
                      << " H=" << H << "/" << config_.minChildWeight << std::endl;
        }
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }
    
    // 寻找最佳分裂
    if (nodeCount <= 5) {
        std::cout << "DEBUG: 开始寻找分裂..." << std::endl;
    }
    
    auto [bestFeature, bestThreshold, bestGain] = 
        xgbFinder_->findBestSplitXGB(X, rowLength, gradients, hessians, indices, *xgbCriterion_);
    
    if (nodeCount <= 5) {
        std::cout << "DEBUG: 分裂结果 - 特征=" << bestFeature 
                  << " 阈值=" << std::fixed << std::setprecision(6) << bestThreshold 
                  << " 增益=" << bestGain << " gamma=" << config_.gamma << std::endl;
    }
    
    // 检查分裂有效性
    if (bestFeature < 0 || bestGain <= 0) {
        if (nodeCount <= 5) {
            std::cout << "DEBUG: 无效分裂，创建叶节点" << std::endl;
        }
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }
    
    // 执行分裂
    node->makeInternal(bestFeature, bestThreshold);
    
    // 分割数据
    std::vector<int> leftIndices, rightIndices;
    for (int idx : indices) {
        if (X[idx * rowLength + bestFeature] <= bestThreshold) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    // 检查分裂后的约束
    double H_left = 0.0, H_right = 0.0;
    for (int idx : leftIndices) H_left += hessians[idx];
    for (int idx : rightIndices) H_right += hessians[idx];
    
    if (H_left < config_.minChildWeight || H_right < config_.minChildWeight ||
        leftIndices.empty() || rightIndices.empty()) {
        
        if (nodeCount <= 5) {
            std::cout << "DEBUG: 分裂后约束失败 - H_left=" 
                      << H_left << " H_right=" << H_right << std::endl;
        }
        node->makeLeaf(leafWeight, leafWeight);
        return;
    }
    
    // 创建子节点
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    
    // 更新指针
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();
    
    if (nodeCount <= 3) {
        std::cout << "DEBUG: 成功分裂！左=" << leftIndices.size() 
                  << " 右=" << rightIndices.size() << std::endl;
    }
    
    // 递归构建子树
    buildXGBNode(node->leftChild.get(), X, rowLength, gradients, hessians, leftIndices, depth + 1);
    buildXGBNode(node->rightChild.get(), X, rowLength, gradients, hessians, rightIndices, depth + 1);
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
        // 不采样，使用全部数据
        return;
    }
    
    // 行采样
    size_t n = gradients.size();
    size_t sampleSize = static_cast<size_t>(n * config_.subsample);
    
    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);
    
    // 随机采样
    std::random_device rd;
    std::mt19937 gen(rd());
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