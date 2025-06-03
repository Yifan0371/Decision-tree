// =============================================================================
// include/lightgbm/trainer/LightGBMTrainer.hpp - 添加并行方法声明
// =============================================================================
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

/** LightGBM训练器 - 深度OpenMP并行优化版本 */
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

    // =============================================
    // 原有方法（保持兼容性）
    // =============================================
    
    void initializeComponents();
    void preprocessFeatures(const std::vector<double>& data,
                            int rowLength,
                            size_t sampleSize);
    double computeBaseScore(const std::vector<double>& y) const;
    std::vector<double> calculateFeatureImportance(int numFeatures) const;

    std::unique_ptr<ISplitCriterion> createCriterion() const;
    std::unique_ptr<ISplitFinder> createOptimalSplitFinder() const;
    std::unique_ptr<ISplitFinder> createHistogramFinder() const;

    double predictSingleTree(const Node* tree,
                             const double* sample,
                             int rowLength) const;
    
    // =============================================
    // 新增：深度并行优化方法
    // =============================================
    
    /** 并行特征预处理 */
    void preprocessFeaturesParallel(const std::vector<double>& data,
                                   int rowLength,
                                   size_t sampleSize);

    /** 并行损失计算 */
    double computeLossParallel(const std::vector<double>& labels,
                              const std::vector<double>& predictions) const;

    /** 并行梯度计算 */
    void computeGradientsParallel(const std::vector<double>& labels,
                                 const std::vector<double>& predictions);

    /** 并行GOSS采样 */
    void performGOSSSamplingParallel();

    /** 并行树构建 */
    std::unique_ptr<Node> buildTreeParallel(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,
        const std::vector<double>& targets,
        const std::vector<int>& sampleIndices,
        const std::vector<double>& sampleWeights,
        const std::vector<FeatureBundle>& bundles) const;

    /** 并行预测更新 */
    void updatePredictionsParallel(const std::vector<double>& data,
                                  int rowLength,
                                  const Node* tree,
                                  std::vector<double>& predictions) const;

    /** 并行早停检查 */
    bool shouldEarlyStopParallel(int currentIter) const;
    
    // =============================================
    // 性能监控和分析工具
    // =============================================
    
    /** 获取GOSS采样效率统计 */
    struct GOSSStats {
        size_t totalSamples;
        size_t selectedSamples;
        double samplingRatio;
        double timeMs;
    };
    
    GOSSStats getGOSSStats() const {
        return {gradients_.size(), sampleIndices_.size(), 
                static_cast<double>(sampleIndices_.size()) / gradients_.size(), 0.0};
    }
    
    /** 获取特征绑定统计 */
    struct BundlingStats {
        int originalFeatures;
        int bundledFeatures;
        double compressionRatio;
    };
    
    BundlingStats getBundlingStats(int originalFeatures) const {
        return {originalFeatures, static_cast<int>(featureBundles_.size()),
                static_cast<double>(originalFeatures) / featureBundles_.size()};
    }
    
    /** 获取并行性能统计 */
    struct ParallelStats {
        double totalTrainingTime;
        double avgIterationTime;
        double parallelEfficiency;
        int threadsUsed;
    };
    
    ParallelStats getParallelStats() const;
    
    /** 估算内存使用量 */
    size_t estimateMemoryUsage() const {
        size_t total = 0;
        total += gradients_.capacity() * sizeof(double);
        total += sampleIndices_.capacity() * sizeof(int);
        total += sampleWeights_.capacity() * sizeof(double);
        total += trainingLoss_.capacity() * sizeof(double);
        return total;
    }
};

#endif // LIGHTGBM_TRAINER_LIGHTGBMTRAINER_HPP