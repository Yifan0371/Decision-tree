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
    
    // 私有方法
    void initializeComponents();
    void preprocessFeatures(const std::vector<double>& data, 
                          int rowLength, 
                          size_t sampleSize);
    double computeBaseScore(const std::vector<double>& y) const;
    std::vector<double> calculateFeatureImportance(int numFeatures) const;
    std::unique_ptr<ISplitFinder> createHistogramFinder() const;
    std::unique_ptr<ISplitCriterion> createCriterion() const;
    
    // **修复: 添加predictSingleTree声明**
    double predictSingleTree(const Node* tree, 
                           const double* sample, 
                           int rowLength) const;
};

#endif // LIGHTGBM_TRAINER_LIGHTGBMTRAINER_HPP