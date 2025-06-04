#pragma once

#include "tree/ISplitFinder.hpp"

class HistogramEWFinder : public ISplitFinder {
public:
    explicit HistogramEWFinder(int bins = 64) : bins_(bins) {}
    
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;

private:
    int bins_;
    
    // **新增**: 优化的传统方法作为备选
    std::tuple<int, double, double> findBestSplitTraditionalOptimized(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const;
};