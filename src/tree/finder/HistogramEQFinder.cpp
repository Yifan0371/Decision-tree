#include "finder/HistogramEQFinder.hpp"
#include <algorithm>
#include <limits>

std::tuple<int,double,double> HistogramEQFinder::findBestSplit(
    const std::vector<double>& X, int D,
    const std::vector<double>& y,
    const std::vector<int>& idx,
    double parentMetric,
    const ISplitCriterion& crit) const
{
    int bestFeat = -1; 
    double bestThr = 0, bestGain = -std::numeric_limits<double>::infinity();
    const int B = bins_;
    
    for (int f = 0; f < D; ++f) {
        std::vector<std::pair<double, int>> pairs;
        pairs.reserve(idx.size());
        for (int i : idx) pairs.emplace_back(X[i*D+f], i);
        std::sort(pairs.begin(), pairs.end());
        
        if (pairs.size() < 2) continue;
        
        int perBucket = std::max(1, static_cast<int>(idx.size()) / B);
        
        for (size_t pivot = perBucket; pivot < pairs.size(); pivot += perBucket) {
            // 确保不会越界
            if (pivot >= pairs.size()) break;
            
            std::vector<int> leftIdx, rightIdx;
            leftIdx.reserve(pivot);
            rightIdx.reserve(pairs.size() - pivot);
            
            for (size_t i = 0; i < pivot; ++i) 
                leftIdx.push_back(pairs[i].second);
            for (size_t i = pivot; i < pairs.size(); ++i) 
                rightIdx.push_back(pairs[i].second);
                
            if (leftIdx.empty() || rightIdx.empty()) continue;
            
            double mL = crit.nodeMetric(y, leftIdx);
            double mR = crit.nodeMetric(y, rightIdx);
            double gain = parentMetric - (mL*leftIdx.size() + mR*rightIdx.size()) / idx.size();
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                // 在相邻值之间设置阈值
                bestThr = 0.5 * (pairs[pivot-1].first + pairs[pivot].first);
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}