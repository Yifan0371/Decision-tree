#pragma once

#include "tree/ISplitFinder.hpp"
#include <string>  

class AdaptiveEWFinder : public ISplitFinder {
public:
    
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
    std::string rule_; 
    
    
    int calculateOptimalBins(const std::vector<double>& values) const;
    double calculateIQR(std::vector<double> values) const;
};
