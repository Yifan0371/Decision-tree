// =============================================================================
// include/xgboost/finder/XGBoostSplitFinder.hpp
// =============================================================================
#ifndef XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP
#define XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP

#include "tree/ISplitFinder.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>    

/** XGBoost分裂查找器：实现基于梯度/Hessian的精确贪婪算法 */
class XGBoostSplitFinder : public ISplitFinder {
public:
    explicit XGBoostSplitFinder(double gamma = 0.0, int minChildWeight = 1)
        : gamma_(gamma), minChildWeight_(minChildWeight) {}
    
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,      // 这里实际存储梯度
        const std::vector<int>& indices,
        double currentMetric,
        const ISplitCriterion& criterion) const override;
    
    /** XGBoost专用分裂查找 */
    std::tuple<int, double, double> findBestSplitXGB(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices,
        const XGBoostCriterion& xgbCriterion) const;

private:
    double gamma_;          // 最小分裂损失
    int minChildWeight_;    // 最小子节点权重和
    
    /** 计算子集的梯度和Hessian统计量 */
    std::pair<double, double> computeGradHessStats(
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices) const;
};

#endif // XGBOOST_FINDER_XGBOOSTSPLITFINDER_HPP