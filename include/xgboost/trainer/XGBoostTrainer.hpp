#pragma once

#include "xgboost/core/XGBoostConfig.hpp"
#include "xgboost/model/XGBoostModel.hpp"
#include "xgboost/loss/XGBoostLossFactory.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"
#include "tree/ITreeTrainer.hpp"
#include <memory>
#include <vector>

// 列存储数据结构 - 优化内存访问
struct ColumnData {
    std::vector<std::vector<int>> sortedIndices;  // 每列的排序索引
    std::vector<double> values;                   // 列存储的值
    int numFeatures;
    size_t numSamples;
    
    ColumnData(int features, size_t samples) : numFeatures(features), numSamples(samples) {
        sortedIndices.resize(features);
        values.reserve(samples * features);
    }
};

class XGBoostTrainer : public ITreeTrainer {
public:
    explicit XGBoostTrainer(const XGBoostConfig& config);

    // ITreeTrainer接口
    void train(const std::vector<double>& data, int rowLength, const std::vector<double>& labels) override;
    double predict(const double* sample, int rowLength) const override;
    void evaluate(const std::vector<double>& X, int rowLength, const std::vector<double>& y, double& mse, double& mae) override;

    // XGBoost专用方法
    const XGBoostModel* getXGBModel() const { return &model_; }
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }
    std::vector<double> getFeatureImportance(int numFeatures) const { return model_.getFeatureImportance(numFeatures); }

    void setValidationData(const std::vector<double>& X_val, const std::vector<double>& y_val, int rowLength) {
        X_val_ = X_val; 
        y_val_ = y_val; 
        valRowLength_ = rowLength; 
        hasValidation_ = true;
    }

private:
    XGBoostConfig config_;
    XGBoostModel model_;
    std::unique_ptr<IRegressionLoss> lossFunction_;
    std::unique_ptr<XGBoostCriterion> xgbCriterion_;

    std::vector<double> trainingLoss_;
    std::vector<double> X_val_, y_val_;
    int valRowLength_ = 0;
    bool hasValidation_ = false;

    // 核心优化方法
    std::unique_ptr<Node> trainSingleTree(const ColumnData& columnData, 
                                         const std::vector<double>& gradients, 
                                         const std::vector<double>& hessians, 
                                         const std::vector<char>& rootMask) const;
    
    void buildXGBNode(Node* node, 
                     const ColumnData& columnData, 
                     const std::vector<double>& gradients,
                     const std::vector<double>& hessians, 
                     const std::vector<char>& nodeMask, 
                     int depth) const;
    
    std::tuple<int, double, double> findBestSplitXGB(
        const ColumnData& columnData,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask) const;
    
    // 辅助方法
    double computeBaseScore(const std::vector<double>& y) const;
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    double computeValidationLoss() const;
    void updatePredictions(const std::vector<double>& data, int rowLength, 
                          const Node* tree, std::vector<double>& predictions) const;
};