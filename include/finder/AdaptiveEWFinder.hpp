#ifndef ADAPTIVE_EW_FINDER_HPP
#define ADAPTIVE_EW_FINDER_HPP
#include "tree/ISplitFinder.hpp"
#include <string>  // 添加这个头文件

class AdaptiveEWFinder : public ISplitFinder {
public:
    // 修复构造函数默认参数
    explicit AdaptiveEWFinder(int minBins = 8, int maxBins = 128, 
                             const std::string& rule = std::string("sturges"))
        : minBins_(minBins), maxBins_(maxBins), rule_(rule) {}
    
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;

private:
    int minBins_;
    int maxBins_;
    std::string rule_; // "sturges", "rice", "freedman_diaconis", "sqrt"
    
    // 计算自适应箱数
    int calculateOptimalBins(const std::vector<double>& values) const;
    double calculateIQR(std::vector<double> values) const;
};
#endif