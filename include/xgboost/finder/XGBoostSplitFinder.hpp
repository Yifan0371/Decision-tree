#pragma once

#include <iostream>          
#include <cmath>
#include <vector>
#include <tuple>
#include <limits>

#include "tree/ISplitFinder.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"


class XGBoostSplitFinder : public ISplitFinder {
public:
    explicit XGBoostSplitFinder(double gamma = 0.0, int minChildWeight = 1)
        : gamma_(gamma), minChildWeight_(minChildWeight) {}

    
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

    
    std::tuple<int, double, double> findBestSplitXGB(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const std::vector<std::vector<int>>& sortedIndicesAll,
        const XGBoostCriterion& xgbCriterion) const;

private:
    double gamma_;          
    int minChildWeight_;    
};
