#include "finder/HistogramEWFinder.hpp"
#include <algorithm>
#include <limits>

std::tuple<int,double,double> HistogramEWFinder::findBestSplit(
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
        double vmin = std::numeric_limits<double>::infinity();
        double vmax = -vmin;
        for (int i : idx) {
            double v = X[i*D+f];
            vmin = std::min(vmin, v); 
            vmax = std::max(vmax, v);
        }
        if (vmax == vmin) continue;
        
        double binWidth = (vmax - vmin) / B;
        std::vector<std::vector<int>> buckets(B);
        
        // 分配样本到桶中
        for (int i : idx) {
            int b = std::min<int>((X[i*D+f] - vmin) / binWidth, B-1);
            buckets[b].push_back(i);
        }
        
        // 优化：使用前缀累积避免重复计算
        std::vector<int> leftIdx;
        leftIdx.reserve(idx.size());
        
        for (int b = 0; b < B-1; ++b) {
            leftIdx.insert(leftIdx.end(), buckets[b].begin(), buckets[b].end());
            if (leftIdx.empty()) continue;
            
            size_t leftSize = leftIdx.size();
            size_t rightSize = idx.size() - leftSize;
            if (rightSize == 0) break;
            
            // 构造右侧索引（优化：只在需要时构造）
            std::vector<int> rightIdx;
            rightIdx.reserve(rightSize);
            for (int k = b+1; k < B; ++k) {
                rightIdx.insert(rightIdx.end(), buckets[k].begin(), buckets[k].end());
            }
            
            double mL = crit.nodeMetric(y, leftIdx);
            double mR = crit.nodeMetric(y, rightIdx);
            double gain = parentMetric - (mL*leftSize + mR*rightSize) / idx.size();
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr = vmin + binWidth * (b + 1);
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}