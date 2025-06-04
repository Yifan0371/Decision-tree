// =============================================================================
// src/xgboost/finder/XGBoostSplitFinder.cpp - 修正版本
// =============================================================================
#include "xgboost/finder/XGBoostSplitFinder.hpp"
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>  // 用于 std::iota

/**
 * 优化版本的 findBestSplitXGB：使用优化的数据结构避免vector<vector>
 */
std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitXGB(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const std::vector<std::vector<int>>& sortedIndicesAll,
    const XGBoostCriterion& xgbCriterion) const {

    const size_t n = nodeMask.size();

    // **优化1: 并行计算当前节点的统计信息**
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    // 检查基本约束
    if (sampleCount < 2 || H_parent < minChildWeight_) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

    // **优化2: 并行遍历特征，避免重复的vector操作**
    #pragma omp parallel if(rowLength > 4)
    {
        // 线程局部变量
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        // 线程局部缓冲区（避免重复分配）
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < rowLength; ++f) {
            // **优化3: 高效构造当前节点在特征f上的有序索引列表**
            nodeSorted.clear();
            
            // 使用引用避免拷贝
            const std::vector<int>& featureIndices = sortedIndicesAll[f];
            
            for (const int idx : featureIndices) {
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

            // **优化4: 单次遍历计算最佳分裂**
            double G_left = 0.0, H_left = 0.0;
            
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = data[idx * rowLength + f];
                const double nextVal = data[nextIdx * rowLength + f];

                // 跳过相同特征值
                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

                // 检查左右子节点的最小 Hessian 约束
                if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;

                // 计算增益
                const double gain = xgbCriterion.computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, gamma_);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
        // **优化5: 线程间归约**
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}

/**
 * 使用优化数据结构的版本
 */
std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitOptimized(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const OptimizedSortedIndices& sortedIndices,
    const XGBoostCriterion& xgbCriterion) const {

    const size_t n = nodeMask.size();

    // 计算当前节点的统计信息
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }

    if (sampleCount < 2 || H_parent < minChildWeight_) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

    // 并行遍历每个特征
    #pragma omp parallel if(rowLength > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < rowLength; ++f) {
            // 获取当前特征的排序索引（避免拷贝）
            const int* featureIndices = sortedIndices.getFeatureData(f);
            const size_t featureSize = sortedIndices.getFeatureSize();
            
            // 构造当前节点在特征f上的有序索引列表
            nodeSorted.clear();
            for (size_t i = 0; i < featureSize; ++i) {
                const int idx = featureIndices[i];
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

            // 遍历分裂点
            double G_left = 0.0, H_left = 0.0;
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = data[idx * rowLength + f];
                const double nextVal = data[nextIdx * rowLength + f];

                // 跳过相同特征值
                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

                // 检查约束
                if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;

                // 计算增益
                const double gain = xgbCriterion.computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, gamma_);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}

/**
 * 批量处理版本，减少函数调用开销
 */
std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitBatch(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const std::vector<std::vector<int>>& sortedIndicesAll,
    const XGBoostCriterion& xgbCriterion,
    const std::vector<int>& candidateFeatures) const {

    const size_t n = nodeMask.size();

    // 计算父节点统计
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }

    if (sampleCount < 2 || H_parent < minChildWeight_) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

    // 只遍历候选特征（用于列采样等场景）
    std::vector<int> featuresToCheck;
    if (candidateFeatures.empty()) {
        featuresToCheck.resize(rowLength);
        std::iota(featuresToCheck.begin(), featuresToCheck.end(), 0);
    } else {
        featuresToCheck = candidateFeatures;
    }

    #pragma omp parallel if(featuresToCheck.size() > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (size_t fi = 0; fi < featuresToCheck.size(); ++fi) {
            const int f = featuresToCheck[fi];
            
            nodeSorted.clear();
            const std::vector<int>& featureIndices = sortedIndicesAll[f];
            
            for (const int idx : featureIndices) {
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

            double G_left = 0.0, H_left = 0.0;
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = data[idx * rowLength + f];
                const double nextVal = data[nextIdx * rowLength + f];

                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

                if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;

                const double gain = xgbCriterion.computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, gamma_);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}