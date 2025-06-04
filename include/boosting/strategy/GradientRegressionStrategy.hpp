#pragma once

#include "../loss/IRegressionLoss.hpp"
#include <memory>
#include <vector>
#include <string>


class GradientRegressionStrategy {
public:
    explicit GradientRegressionStrategy(
        std::unique_ptr<IRegressionLoss> lossFunc,
        double baseLearningRate = 0.1,
        bool useLineSearch = false)
        : lossFunc_(std::move(lossFunc)),
          baseLearningRate_(baseLearningRate),
          useLineSearch_(useLineSearch) {}

    
    void updateTargets(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& targets) const;

    
    double computeLearningRate(
        int ,
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const {
        if (!useLineSearch_) {
            return baseLearningRate_;
        }
        
        return computeOptimalLearningRate(y_true, y_pred, tree_pred);
    }

    
    void updatePredictions(
        const std::vector<double>& tree_pred,
        double learning_rate,
        std::vector<double>& y_pred) const;

    std::string name() const { return "gradient_regression"; }

    const IRegressionLoss* getLossFunction() const { return lossFunc_.get(); }

    
    double computeTotalLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const;

private:
    std::unique_ptr<IRegressionLoss> lossFunc_;
    double baseLearningRate_;
    bool useLineSearch_;

    
    double computeOptimalLearningRate(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const;

    
    double evaluateLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred,
        double lr) const;
};
