#ifndef ADAPTIVE_EQ_FINDER_HPP
#define ADAPTIVE_EQ_FINDER_HPP
#include "tree/ISplitFinder.hpp"

class AdaptiveEQFinder : public ISplitFinder {
public:
    explicit AdaptiveEQFinder(int minSamplesPerBin = 5, int maxBins = 64,
                             double variabilityThreshold = 0.1)
        : minSamplesPerBin_(minSamplesPerBin), maxBins_(maxBins),
          variabilityThreshold_(variabilityThreshold) {}
    
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;

private:
    int minSamplesPerBin_;
    int maxBins_;
    double variabilityThreshold_;
    
    // 计算最优等频分箱参数
    std::pair<int, int> calculateOptimalFrequencyParams(
        const std::vector<double>& values) const;
    
    double calculateVariability(const std::vector<double>& values) const;
};
#endif
