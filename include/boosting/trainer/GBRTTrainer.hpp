#pragma once

#include "../model/RegressionBoostingModel.hpp"
#include "../strategy/GradientRegressionStrategy.hpp"
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "../dart/IDartStrategy.hpp"
#include <memory>
#include <iostream>
#include <vector>
#include <random>


struct GBRTConfig {
    
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    
    
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    
    
    double subsample = 1.0;
    
    
    bool useLineSearch = false;
    
    
    bool enableDart = false;
    double dartDropRate = 0.1;
    bool dartNormalize = true;
    bool dartSkipDropForPrediction = false;
    std::string dartStrategy = "uniform";
    uint32_t dartSeed = 42;
    std::string dartWeightStrategy = "mild";
};


class GBRTTrainer {
public:
    explicit GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy);
    
    void train(const std::vector<double>& X,
               int rowLength,
               const std::vector<double>& y);
    
    double predict(const double* sample, int rowLength) const;
    
    std::vector<double> predictBatch(
        const std::vector<double>& X, int rowLength) const;
    
    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& loss,
                  double& mse,
                  double& mae);
    
    const RegressionBoostingModel* getModel() const { return &model_; }
    std::string name() const { return "GBRT"; }
    
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }
    
    
    std::vector<double> getFeatureImportance(int numFeatures) const {
        return model_.getFeatureImportance(numFeatures);
    }
    
    
    void setValidationData(const std::vector<double>& X_val,
                          const std::vector<double>& y_val,
                          int rowLength) {
        X_val_ = X_val;
        y_val_ = y_val;
        valRowLength_ = rowLength;
        hasValidation_ = true;
    }

private:
    GBRTConfig config_;
    std::unique_ptr<GradientRegressionStrategy> strategy_;
    RegressionBoostingModel model_;
    std::vector<double> trainingLoss_;
    std::vector<double> validationLoss_;
    
    
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;
    
    
    std::unique_ptr<IDartStrategy> dartStrategy_;
    mutable std::mt19937 dartGen_;
    
    
    std::unique_ptr<SingleTreeTrainer> createTreeTrainer() const;
    
    
    double computeBaseScore(const std::vector<double>& y) const;
    
    
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    
    
    void sampleData(const std::vector<double>& X, int rowLength,
                   const std::vector<double>& targets,
                   std::vector<double>& sampledX,
                   std::vector<double>& sampledTargets) const;
    
    
    std::unique_ptr<Node> cloneTree(const Node* original) const;
    
    
    double computeValidationLoss(const std::vector<double>& predictions) const;
    
    
    std::unique_ptr<IDartStrategy> createDartStrategy() const;
    
    
    void trainStandard(const std::vector<double>& X,
                      int rowLength,
                      const std::vector<double>& y);
    
    
    void trainWithDart(const std::vector<double>& X,
                      int rowLength,
                      const std::vector<double>& y);
    
    
    void updatePredictionsWithDropout(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<int>& droppedTrees,
        std::vector<double>& predictions) const;
    
    
    
    
    
    
    void batchTreePredict(const SingleTreeTrainer* trainer,
                         const std::vector<double>& X,
                         int rowLength,
                         std::vector<double>& predictions) const;
    
    
    void batchModelPredict(const std::vector<double>& X,
                          int rowLength,
                          std::vector<double>& predictions) const;
    
    
    void updatePredictionsWithDropoutParallel(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<int>& droppedTrees,
        std::vector<double>& predictions) const;
};
