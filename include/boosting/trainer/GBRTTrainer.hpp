// =============================================================================
// include/boosting/trainer/GBRTTrainer.hpp
// =============================================================================
#ifndef BOOSTING_TRAINER_GBRTTRAINER_HPP
#define BOOSTING_TRAINER_GBRTTRAINER_HPP

#include "../model/RegressionBoostingModel.hpp"
#include "../strategy/GradientRegressionStrategy.hpp"
#include "tree/trainer/SingleTreeTrainer.hpp"
#include <memory>
#include <iostream>
#include <vector>

/** 梯度提升回归树配置 */
struct GBRTConfig {
    // 核心参数
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    
    // 决策树参数（复用你现有的分割器和准则）
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    // 训练控制
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    
    // 正则化参数
    double subsample = 1.0;            // 样本采样比例
    
    // 性能优化
    bool useLineSearch = false;        // 是否使用线搜索优化学习率
};

/** 梯度提升回归树训练器 */
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
    
    /** 获取特征重要性 */
    std::vector<double> getFeatureImportance(int numFeatures) const {
        return model_.getFeatureImportance(numFeatures);
    }
    
    /** GBRT特有方法：设置验证集用于早停 */
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
    
    // 验证集数据
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;
    
    /** 创建单树训练器（复用你现有的SingleTreeTrainer） */
    std::unique_ptr<SingleTreeTrainer> createTreeTrainer() const;
    
    /** 计算基准分数（回归任务通常用均值） */
    double computeBaseScore(const std::vector<double>& y) const;
    
    /** 检查早停条件 */
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    
    /** 样本采样 */
    void sampleData(const std::vector<double>& X, int rowLength,
                   const std::vector<double>& targets,
                   std::vector<double>& sampledX,
                   std::vector<double>& sampledTargets) const;
    
    /** 深度复制树节点 */
    std::unique_ptr<Node> cloneTree(const Node* original) const;
    
    /** 计算验证集损失 */
    double computeValidationLoss(const std::vector<double>& predictions) const;
};

#endif // BOOSTING_TRAINER_GBRTTRAINER_HPP