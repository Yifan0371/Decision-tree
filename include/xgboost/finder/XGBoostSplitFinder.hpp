// =============================================================================
// include/xgboost/finder/XGBoostSplitFinder.hpp - 修正版本
// =============================================================================
#pragma once

#include <iostream>          
#include <cmath>
#include <vector>
#include <tuple>
#include <limits>

#include "tree/ISplitFinder.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"

// 完整定义优化的数据结构（替代前向声明）
struct OptimizedSortedIndices {
    std::vector<int> data;      // 连续存储所有索引
    std::vector<size_t> offsets; // 每个特征的起始偏移
    int numFeatures;
    size_t numSamples;
    
    OptimizedSortedIndices(int features, size_t samples) 
        : numFeatures(features), numSamples(samples) {
        data.resize(features * samples);
        offsets.resize(features + 1);
        for (int f = 0; f <= features; ++f) {
            offsets[f] = f * samples;
        }
    }
    
    // 获取特征f的索引范围
    std::pair<int*, int*> getFeatureRange(int f) {
        int* start = data.data() + offsets[f];
        int* end = data.data() + offsets[f + 1];
        return {start, end};
    }
    
    const int* getFeatureData(int f) const {
        return data.data() + offsets[f];
    }
    
    size_t getFeatureSize() const { return numSamples; }
};

class XGBoostSplitFinder : public ISplitFinder {
public:
    explicit XGBoostSplitFinder(double gamma = 0.0, int minChildWeight = 1)
        : gamma_(gamma), minChildWeight_(minChildWeight) {}

    // **保留旧接口的兼容性**
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& /*data*/,
        int /*rowLength*/,
        const std::vector<double>& /*labels*/,
        const std::vector<int>& /*indices*/,
        double /*currentMetric*/,
        const ISplitCriterion& /*criterion*/) const override {
        std::cout << "WARNING: 旧接口被调用，应使用 findBestSplitXGB" << std::endl;
        return {-1, 0.0, 0.0};
    }

    // **原版XGB分裂查找方法**
    std::tuple<int, double, double> findBestSplitXGB(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const std::vector<std::vector<int>>& sortedIndicesAll,
        const XGBoostCriterion& xgbCriterion) const;

    // **优化版本：使用优化的数据结构**
    std::tuple<int, double, double> findBestSplitOptimized(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const OptimizedSortedIndices& sortedIndices,
        const XGBoostCriterion& xgbCriterion) const;

    // **批量处理版本：支持特征子集**
    std::tuple<int, double, double> findBestSplitBatch(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const std::vector<std::vector<int>>& sortedIndicesAll,
        const XGBoostCriterion& xgbCriterion,
        const std::vector<int>& candidateFeatures = {}) const;

    // **获取参数**
    double getGamma() const { return gamma_; }
    int getMinChildWeight() const { return minChildWeight_; }

    // **设置参数**
    void setGamma(double gamma) { gamma_ = gamma; }
    void setMinChildWeight(int minChildWeight) { minChildWeight_ = minChildWeight; }

private:
    double gamma_;          // 最小分裂增益
    int minChildWeight_;    // 子节点最小Hessian和
};