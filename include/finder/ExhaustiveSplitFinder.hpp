/*  ExhaustiveSplitFinder.hpp  */
#ifndef EXHAUSTIVE_SPLIT_FINDER_HPP
#define EXHAUSTIVE_SPLIT_FINDER_HPP

#include "../tree/ISplitFinder.hpp"
#include <vector>
#include <tuple>

class ExhaustiveSplitFinder : public ISplitFinder {
public:
    /**
     * @brief 在给定样本索引集合上搜索最优切分
     * @param data        按行存放的特征值数组，长度 = 样本数 * rowLength
     * @param rowLength   每个样本的特征维数
     * @param labels      样本标签
     * @param indices     当前节点包含的样本索引
     * @param currentMetric 父节点的评估指标（如 MSE）
     * @param criterion   切分准则（此处仅用到回调接口）
     * @return (最佳特征下标, 切分阈值, 信息增益)
     */
    std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int                         rowLength,
                  const std::vector<double>&  labels,
                  const std::vector<int>&     indices,
                  double                      currentMetric,
                  const ISplitCriterion&      criterion) const override;
};

#endif /* EXHAUSTIVE_SPLIT_FINDER_HPP */