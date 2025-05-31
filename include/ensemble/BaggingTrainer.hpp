#ifndef ENSEMBLE_BAGGING_TRAINER_HPP
#define ENSEMBLE_BAGGING_TRAINER_HPP

#include "../tree/ITreeTrainer.hpp"
#include "../tree/ISplitFinder.hpp"
#include "../tree/ISplitCriterion.hpp"
#include "../tree/IPruner.hpp"
#include "tree/trainer/SingleTreeTrainer.hpp"
#include <vector>
#include <memory>
#include <random>

/**
 * Bootstrap Aggregating (Bagging) 训练器
 * 训练多个决策树并在预测时取平均值
 */
class BaggingTrainer : public ITreeTrainer {
public:
    /**
     * 构造函数
     * @param numTrees 树的数量
     * @param sampleRatio Bootstrap采样比例 (0.0, 1.0]，默认1.0表示与原数据集同样大小
     * @param maxDepth 每棵树的最大深度
     * @param minSamplesLeaf 每棵树叶节点的最小样本数
     * @param criterion 分割准则名称
     * @param splitMethod 分割方法名称
     * @param prunerType 剪枝类型
     * @param prunerParam 剪枝参数
     * @param seed 随机种子
     */
    BaggingTrainer(int numTrees = 10,
                   double sampleRatio = 1.0,
                   int maxDepth = 800,
                   int minSamplesLeaf = 2,
                   const std::string& criterion = "mse",
                   const std::string& splitMethod = "exhaustive",
                   const std::string& prunerType = "none",
                   double prunerParam = 0.01,
                   uint32_t seed = 42);

    // 实现 ITreeTrainer 接口
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

    // Bagging特有的方法
    int getNumTrees() const { return numTrees_; }
    double getSampleRatio() const { return sampleRatio_; }
    
    /**
     * 获取特征重要性（基于所有树的平均）
     * @param numFeatures 特征数量
     * @return 特征重要性向量
     */
    std::vector<double> getFeatureImportance(int numFeatures) const;

    /**
     * 获取袋外误差（Out-of-Bag Error）
     * @param data 训练数据
     * @param rowLength 特征数量
     * @param labels 训练标签
     * @return OOB MSE
     */
    double getOOBError(const std::vector<double>& data,
                       int rowLength,
                       const std::vector<double>& labels) const;

private:
    // 参数
    int numTrees_;
    double sampleRatio_;
    int maxDepth_;
    int minSamplesLeaf_;
    std::string criterion_;
    std::string splitMethod_;
    std::string prunerType_;
    double prunerParam_;
    
    // 随机数生成器
    mutable std::mt19937 gen_;
    
    // 存储的训练器和采样信息
    std::vector<std::unique_ptr<SingleTreeTrainer>> trees_;
    std::vector<std::vector<int>> oobIndices_;  // 每棵树的袋外样本索引
    
    // 工厂方法：创建分割器、准则、剪枝器
    std::unique_ptr<ISplitFinder> createSplitFinder() const;
    std::unique_ptr<ISplitCriterion> createCriterion() const;
    std::unique_ptr<IPruner> createPruner(const std::vector<double>& X_val,
                                         int rowLength,
                                         const std::vector<double>& y_val) const;
    
    /**
     * Bootstrap采样（优化版本）
     * @param dataSize 原始数据大小
     * @param sampleIndices 输出：采样索引
     * @param oobIndices 输出：袋外索引
     */
    void bootstrapSample(int dataSize,
                        std::vector<int>& sampleIndices,
                        std::vector<int>& oobIndices) const;
    
    /**
     * 根据索引提取子数据集（优化版本，减少内存复制）
     */
    void extractSubsetOptimized(const std::vector<double>& originalData,
                               int rowLength,
                               const std::vector<double>& originalLabels,
                               const std::vector<int>& indices,
                               std::vector<double>& subData,
                               std::vector<double>& subLabels) const;
};

#endif // ENSEMBLE_BAGGING_TRAINER_HPP