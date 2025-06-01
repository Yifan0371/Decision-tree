#ifndef LIGHTGBM_TRAINER_LIGHTGBMTRAINER_HPP
#define LIGHTGBM_TRAINER_LIGHTGBMTRAINER_HPP

#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/model/LightGBMModel.hpp"
#include "lightgbm/sampling/GOSSSampler.hpp"
#include "lightgbm/feature/FeatureBundler.hpp"
#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include "boosting/loss/IRegressionLoss.hpp"
#include "tree/ITreeTrainer.hpp"
#include <memory>
#include <vector>

/** LightGBM训练器 */
class LightGBMTrainer : public ITreeTrainer {
public:
    explicit LightGBMTrainer(const LightGBMConfig& config);
    
    // ITreeTrainer接口
    void train(const std::vector<double>& data,
               int rowLength,
               const std::vector<double>& labels) override;
               
    double predict(const double* sample, int rowLength) const override;
    
    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae) override;
    
    // LightGBM专用方法
    const LightGBMModel* getLGBModel() const { return &model_; }
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }
    
    std::vector<double> getFeatureImportance(int numFeatures) const {
        return calculateFeatureImportance(numFeatures);
    }

private:
    LightGBMConfig config_;
    LightGBMModel model_;
    std::unique_ptr<IRegressionLoss> lossFunction_;
    std::unique_ptr<GOSSSampler> gossSampler_;
    std::unique_ptr<FeatureBundler> featureBundler_;
    std::unique_ptr<LeafwiseTreeBuilder> treeBuilder_;
    
    std::vector<double> trainingLoss_;
    std::vector<FeatureBundle> featureBundles_;
    
    // 内存池（预分配，避免重复分配）
    mutable std::vector<double> gradients_;
    mutable std::vector<int> sampleIndices_;
    mutable std::vector<double> sampleWeights_;
    
    // 初始化组件
    void initializeComponents();
    
    // 预处理特征绑定
    void preprocessFeatures(const std::vector<double>& data, 
                          int rowLength, 
                          size_t sampleSize);
    
    // 计算基准分数
    double computeBaseScore(const std::vector<double>& y) const;
    
    // 计算特征重要性
    std::vector<double> calculateFeatureImportance(int numFeatures) const;
    
    // 创建直方图分割器（复用现有实现）
    std::unique_ptr<ISplitFinder> createHistogramFinder() const;
    
    // 创建准则（复用现有实现）
    std::unique_ptr<ISplitCriterion> createCriterion() const;
};

#endif // LIGHTGBM_TRAINER_LIGHTGBMTRAINER_HPP
