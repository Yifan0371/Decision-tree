// =============================================================================
// include/boosting/trainer/GBRTTrainer.hpp - 深度并行优化版本
// =============================================================================
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
    // 基本参数
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    
    // 树构建参数
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    // 训练控制参数
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    
    // 采样参数
    double subsample = 1.0;
    
    // 线搜索参数
    bool useLineSearch = false;
    
    // DART参数
    bool enableDart = false;
    double dartDropRate = 0.1;
    bool dartNormalize = true;
    bool dartSkipDropForPrediction = false;
    std::string dartStrategy = "uniform";
    uint32_t dartSeed = 42;
    std::string dartWeightStrategy = "mild";
    
    // **新增: 并行优化参数**
    int parallelThreshold = 1000;      // 启用并行的最小样本数
    int chunkSize = 2048;              // 并行chunk大小
    bool enableVectorization = true;   // 启用向量化优化
    bool enableMemoryPool = true;      // 启用内存池
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
    std::string name() const { return "GBRT_Optimized"; }
    
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
    
    // 验证数据
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;
    
    // DART组件
    std::unique_ptr<IDartStrategy> dartStrategy_;
    mutable std::mt19937 dartGen_;
    
    // **核心优化方法**
    void trainStandardOptimized(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y);
    
    void trainWithDartOptimized(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y);
    
    // **并行计算方法**
    double computeBaseScoreParallel(const std::vector<double>& y) const;
    
    double computeTotalLossParallel(const std::vector<double>& y,
                                   const std::vector<double>& pred) const;
    
    void computeResidualsParallel(const std::vector<double>& y,
                                 const std::vector<double>& pred,
                                 std::vector<double>& residuals) const;
    
    void batchTreePredictOptimized(const SingleTreeTrainer* trainer,
                                  const std::vector<double>& X,
                                  int rowLength,
                                  std::vector<double>& predictions) const;
    
    void updatePredictionsVectorized(const std::vector<double>& treePred,
                                    double lr,
                                    std::vector<double>& predictions) const;
    
    // **DART专用并行方法**
    void computeDartPredictionsParallel(const std::vector<double>& X,
                                       int rowLength,
                                       const std::vector<int>& droppedTrees,
                                       std::vector<double>& predictions) const;
    
    void recomputeFullPredictionsParallel(const std::vector<double>& X,
                                         int rowLength,
                                         std::vector<double>& predictions) const;
    
    // **优化辅助方法**
    inline double predictSingleTreeFast(const Node* tree, const double* sample) const;
    
    std::unique_ptr<Node> cloneTreeOptimized(const Node* original) const;
    
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    
    // **原有方法保留**
    std::unique_ptr<SingleTreeTrainer> createTreeTrainer() const;
    std::unique_ptr<IDartStrategy> createDartStrategy() const;
    double computeValidationLoss(const std::vector<double>& predictions) const;
    
    // **弃用的旧方法 - 保留兼容性**
    void sampleData(const std::vector<double>& /* X */, int /* rowLength */,
                   const std::vector<double>& /* targets */,
                   std::vector<double>& /* sampledX */,
                   std::vector<double>& /* sampledTargets */) const {}
    
    std::unique_ptr<Node> cloneTree(const Node* original) const {
        return cloneTreeOptimized(original);
    }
};