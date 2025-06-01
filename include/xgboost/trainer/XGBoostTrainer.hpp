#ifndef XGBOOST_TRAINER_XGBOOSTTRAINER_HPP
#define XGBOOST_TRAINER_XGBOOSTTRAINER_HPP

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

/**
 * XGBoostTrainer：实现完整的 XGBoost 算法，
 * 已改造为“预排序 + 掩码过滤”版本。
 */
class XGBoostTrainer : public ITreeTrainer {
public:
    explicit XGBoostTrainer(const XGBoostConfig& config);

    // ITreeTrainer 接口
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

    // XGBoost 专用方法
    const XGBoostModel* getXGBModel() const { return &model_; }
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }

    /** 设置验证集用于早停 */
    void setValidationData(const std::vector<double>& X_val,
                           const std::vector<double>& y_val,
                           int rowLength) {
        X_val_ = X_val;
        y_val_ = y_val;
        valRowLength_ = rowLength;
        hasValidation_ = true;
    }

    /** 获取特征重要性 */
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

    // 验证集数据
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;

    /** 训练单棵 XGBoost 树（已改造），传入根节点掩码和全局预排序结果 */
    std::unique_ptr<Node> trainSingleTree(
        const std::vector<double>& X,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& rootMask,
        const std::vector<std::vector<int>>& sortedIndicesAll) const;

    /** 递归构建 XGBoost 树节点（已改造） */
    void buildXGBNode(Node* node,
                      const std::vector<double>& X,
                      int rowLength,
                      const std::vector<double>& gradients,
                      const std::vector<double>& hessians,
                      const std::vector<char>& nodeMask,
                      const std::vector<std::vector<int>>& sortedIndicesAll,
                      int depth) const;

    /** 计算基准分数 */
    double computeBaseScore(const std::vector<double>& y) const;

    /** 数据采样（支持行采样），不变 */
    void sampleData(const std::vector<double>& X,
                    int rowLength,
                    const std::vector<double>& gradients,
                    const std::vector<double>& hessians,
                    std::vector<int>& sampleIndices,
                    std::vector<int>& featureIndices) const;

    /** 检查早停条件 */
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;

    /** 计算验证集损失 */
    double computeValidationLoss() const;
};

#endif // XGBOOST_TRAINER_XGBOOSTTRAINER_HPP
