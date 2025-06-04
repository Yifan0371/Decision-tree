// =============================================================================
// include/xgboost/trainer/XGBoostTrainer.hpp - 优化版本
// =============================================================================
#pragma once

#include "xgboost/core/XGBoostConfig.hpp"
#include "xgboost/model/XGBoostModel.hpp"
#include "xgboost/loss/XGBoostLossFactory.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"
#include "xgboost/finder/XGBoostSplitFinder.hpp"  // 这里已经包含了 OptimizedSortedIndices 的定义
#include "tree/ITreeTrainer.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

// 前向声明优化的数据结构（实际定义在 XGBoostSplitFinder.hpp 中）
// struct OptimizedSortedIndices;    // 可保留也可去掉，因 XGBoostSplitFinder.hpp 已定义

class XGBoostTrainer : public ITreeTrainer {
public:
    explicit XGBoostTrainer(const XGBoostConfig& config);

    // ITreeTrainer 接口
    void train(const std::vector<double>& data,
               int rowLength,
               const std::vector<double>& labels) override;

    double predict(const double* sample,
                   int rowLength) const override;

    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae) override;

    // XGBoost 专用方法
    const XGBoostModel* getXGBModel() const { return &model_; }
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }

    void setValidationData(const std::vector<double>& X_val,
                           const std::vector<double>& y_val,
                           int rowLength) {
        X_val_ = X_val;
        y_val_ = y_val;
        valRowLength_ = rowLength;
        hasValidation_ = true;
    }

    std::vector<double> getFeatureImportance(int numFeatures) const {
        return model_.getFeatureImportance(numFeatures);
    }

private:
    XGBoostConfig config_;
    XGBoostModel model_;
    std::unique_ptr<IRegressionLoss> lossFunction_;
    std::unique_ptr<XGBoostCriterion> xgbCriterion_;
    std::unique_ptr<XGBoostSplitFinder> xgbFinder_;

    std::vector<double> trainingLoss_;
    std::vector<double> validationLoss_;

    // 验证数据
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_ = 0;
    bool hasValidation_ = false;

    // **优化后的核心方法（避免vector<vector>）**
    std::unique_ptr<Node> trainSingleTreeOptimized(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& rootMask,
        const OptimizedSortedIndices& sortedIndices) const;

    void buildXGBNodeOptimized(Node* node,
                              const std::vector<double>& X,
                              int rowLength,
                              const std::vector<double>& gradients,
                              const std::vector<double>& hessians,
                              const std::vector<char>& nodeMask,
                              const OptimizedSortedIndices& sortedIndices,
                              int depth) const;

    std::tuple<int, double, double> findBestSplitOptimized(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const OptimizedSortedIndices& sortedIndices) const;

    // **保留旧接口的兼容性方法**
    std::unique_ptr<Node> trainSingleTree(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& rootMask,
        const std::vector<std::vector<int>>& sortedIndicesAll) const;

    void buildXGBNode(Node* node,
                      const std::vector<double>& X,
                      int rowLength,
                      const std::vector<double>& gradients,
                      const std::vector<double>& hessians,
                      const std::vector<char>& nodeMask,
                      const std::vector<std::vector<int>>& sortedIndicesAll,
                      int depth) const;

    // **辅助方法**
    double computeBaseScore(const std::vector<double>& y) const;

    void sampleData(const std::vector<double>& X,
                    int rowLength,
                    const std::vector<double>& gradients,
                    const std::vector<double>& hessians,
                    std::vector<int>& sampleIndices,
                    std::vector<int>& featureIndices) const;

    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    double computeValidationLoss() const;

    void updatePredictionsParallel(const std::vector<double>& data,
                                  int rowLength,
                                  const Node* tree,
                                  std::vector<double>& predictions) const;

    bool shouldConverge(const std::vector<double>& gradients) const;

    // **性能监控方法**
    size_t estimateMemoryUsage(size_t sampleCount, int featureCount) const {
        return sampleCount * (featureCount * sizeof(int) + 4 * sizeof(double) + sizeof(char));
    }

    struct ParallelStats {
        double totalTime;
        double parallelTime;
        double serialTime;
        double efficiency;
    };

    ParallelStats getParallelStats() const {
        return {0.0, 0.0, 0.0, 0.0};
    }
};
