#pragma once

#include "xgboost/core/XGBoostConfig.hpp"
#include "xgboost/model/XGBoostModel.hpp"
#include "xgboost/loss/XGBoostLossFactory.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"
#include "xgboost/finder/XGBoostSplitFinder.hpp"
#include "tree/ITreeTrainer.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>


class XGBoostTrainer : public ITreeTrainer {
public:
    explicit XGBoostTrainer(const XGBoostConfig& config);

    
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

    
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;

    
    
    
    
    
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

    
    double computeBaseScore(const std::vector<double>& y) const;

    
    void sampleData(const std::vector<double>& X,
                    int rowLength,
                    const std::vector<double>& gradients,
                    const std::vector<double>& hessians,
                    std::vector<int>& sampleIndices,
                    std::vector<int>& featureIndices) const;

    
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;

    
    double computeValidationLoss() const;
    
    
    
    
    
    
    std::unique_ptr<Node> trainSingleTreeParallel(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& rootMask,
        const std::vector<std::vector<int>>& sortedIndicesAll) const;

    
    void buildXGBNodeParallel(Node* node,
                             const std::vector<double>& X,
                             int rowLength,
                             const std::vector<double>& gradients,
                             const std::vector<double>& hessians,
                             const std::vector<char>& nodeMask,
                             const std::vector<std::vector<int>>& sortedIndicesAll,
                             int depth) const;

    
    void updatePredictionsParallel(const std::vector<double>& data,
                                  int rowLength,
                                  const Node* tree,
                                  std::vector<double>& predictions) const;

    
    bool shouldConverge(const std::vector<double>& gradients) const;
    
    
    
    
    
    
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
