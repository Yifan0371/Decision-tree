#ifndef HISTOGRAM_EQ_FINDER_HPP
#define HISTOGRAM_EQ_FINDER_HPP
#include "tree/ISplitFinder.hpp"

class HistogramEQFinder : public ISplitFinder {
public:
    explicit HistogramEQFinder(int bins = 64) : bins_(bins) {}
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;
private:
    int bins_;
};
#endif