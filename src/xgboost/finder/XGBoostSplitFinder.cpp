#include "xgboost/finder/XGBoostSplitFinder.hpp"
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>

/**
 * 新版 findBestSplitXGB 实现：利用预排序结果 + 掩码过滤，避免每节点重复排序
 */
std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitXGB(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const std::vector<std::vector<int>>& sortedIndicesAll,
    const XGBoostCriterion& xgbCriterion) const {

    size_t n = nodeMask.size();  // 全局样本总数

    // 1. 计算当前节点的 G_parent, H_parent 及样本数量
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    // 如果样本数不足或 H_parent < minChildWeight，则无法分裂
    if (sampleCount < 2 || H_parent < minChildWeight_) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    const double EPS = 1e-12;

    // 2. 遍历每个特征 f，利用预排序 + 掩码过滤构造 nodeSorted
    for (int f = 0; f < rowLength; ++f) {
        // 2.1 构造当前节点在特征 f 上的有序索引列表 nodeSorted
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        for (int idx : sortedIndicesAll[f]) {
            if (nodeMask[idx]) {
                nodeSorted.push_back(idx);
            }
        }
        if (nodeSorted.size() < 2) {
            continue;
        }

        // 2.2 遍历 nodeSorted 找分裂点
        double G_left = 0.0, H_left = 0.0;
        for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
            int idx = nodeSorted[i];
            G_left += gradients[idx];
            H_left += hessians[idx];

            int nextIdx = nodeSorted[i + 1];
            double currentVal = data[idx * rowLength + f];
            double nextVal = data[nextIdx * rowLength + f];

            // 跳过相同特征值
            if (std::abs(nextVal - currentVal) < EPS) continue;

            double G_right = G_parent - G_left;
            double H_right = H_parent - H_left;

            // 检查左右子节点的最小 Hessian 约束
            if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;

            // 计算增益
            double gain = xgbCriterion.computeSplitGain(
                G_left, H_left,
                G_right, H_right,
                G_parent, H_parent,
                gamma_);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = f;
                bestThreshold = 0.5 * (currentVal + nextVal);
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}
