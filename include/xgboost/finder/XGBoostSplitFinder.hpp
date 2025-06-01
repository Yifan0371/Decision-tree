#ifndef XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP
#define XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP

#include <iostream>          // 为 std::cout, std::endl 提供声明
#include <cmath>
#include <vector>
#include <tuple>
#include <limits>

#include "tree/ISplitFinder.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"

/**
 * XGBoostSplitFinder：基于梯度/Hessian 的精确贪婪算法，
 * 已改造为“预排序 + 掩码过滤”版本，不再在每个节点重复排序。
 */
class XGBoostSplitFinder : public ISplitFinder {
public:
    explicit XGBoostSplitFinder(double gamma = 0.0, int minChildWeight = 1)
        : gamma_(gamma), minChildWeight_(minChildWeight) {}

    // 保留旧接口，仅打印警告，不真正使用
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,
        const std::vector<int>& indices,
        double currentMetric,
        const ISplitCriterion& criterion) const override {
        std::cout << "WARNING: 旧接口被调用，应使用 findBestSplitXGB" << std::endl;
        return {-1, 0.0, 0.0};
    }

    /**
     * 新接口：findBestSplitXGB
     *   - data:       原始特征矩阵（行优先存储）
     *   - rowLength:  每个样本的维度（特征数）
     *   - gradients:  当前节点所有样本的梯度向量
     *   - hessians:   当前节点所有样本的 Hessian 向量
     *   - nodeMask:   长度 = 样本总数 的 0/1 掩码，1 表示该全局索引样本属于本节点
     *   - sortedIndicesAll: 长度 = rowLength 的二维数组，
     *        sortedIndicesAll[f] 是全局对第 f 个特征做一次升序排序后的全体样本索引列表
     *   - xgbCriterion: XGBoostCriterion 对象，用于计算增益等
     *
     * 返回值：{bestFeature, bestThreshold, bestGain}
     */
    std::tuple<int, double, double> findBestSplitXGB(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const std::vector<std::vector<int>>& sortedIndicesAll,
        const XGBoostCriterion& xgbCriterion) const;

private:
    double gamma_;          // 最小分裂增益阈值 gamma
    int minChildWeight_;    // 最小子节点 Hessian 权重和
};

#endif // XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP
