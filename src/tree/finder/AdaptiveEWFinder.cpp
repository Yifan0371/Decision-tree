#include "finder/AdaptiveEWFinder.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

int AdaptiveEWFinder::calculateOptimalBins(const std::vector<double>& values) const {
    int n = static_cast<int>(values.size());
    if (n <= 1) return 1;
    
    int bins = minBins_;
    
    if (rule_ == "sturges") {
        // Sturges规则: k = ⌈log₂(n)⌉ + 1
        bins = static_cast<int>(std::ceil(std::log2(n))) + 1;
    }
    else if (rule_ == "rice") {
        // Rice规则: k = ⌈2 * ∛n⌉
        bins = static_cast<int>(std::ceil(2.0 * std::cbrt(n)));
    }
    else if (rule_ == "sqrt") {
        // 平方根规则: k = ⌈√n⌉
        bins = static_cast<int>(std::ceil(std::sqrt(n)));
    }
    else if (rule_ == "freedman_diaconis") {
        // Freedman-Diaconis规则: h = 2*IQR/∛n, k = (max-min)/h
        double iqr = calculateIQR(values);
        if (iqr > 0) {
            auto minmax = std::minmax_element(values.begin(), values.end());
            double range = *minmax.second - *minmax.first;
            double h = 2.0 * iqr / std::cbrt(n);
            bins = static_cast<int>(std::ceil(range / h));
        }
    }
    
    // 限制在合理范围内
    return std::max(minBins_, std::min(maxBins_, bins));
}

double AdaptiveEWFinder::calculateIQR(std::vector<double> values) const {
    if (values.size() < 4) return 0.0;
    
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    
    // 计算Q1和Q3
    size_t q1_idx = n / 4;
    size_t q3_idx = 3 * n / 4;
    
    return values[q3_idx] - values[q1_idx];
}

std::tuple<int,double,double> AdaptiveEWFinder::findBestSplit(
    const std::vector<double>& X, int D,
    const std::vector<double>& y,
    const std::vector<int>& idx,
    double parentMetric,
    const ISplitCriterion& crit) const
{
    int bestFeat = -1; 
    double bestThr = 0, bestGain = -std::numeric_limits<double>::infinity();
    
    for (int f = 0; f < D; ++f) {
        // 提取特征值
        std::vector<double> values;
        values.reserve(idx.size());
        for (int i : idx) values.push_back(X[i*D+f]);
        
        // 计算自适应箱数
        int B = calculateOptimalBins(values);
        
        double vmin = *std::min_element(values.begin(), values.end());
        double vmax = *std::max_element(values.begin(), values.end());
        if (vmax == vmin) continue;
        
        double binWidth = (vmax - vmin) / B;
        std::vector<std::vector<int>> buckets(B);
        
        // 分配样本到桶中
        for (int i : idx) {
            int b = std::min<int>((X[i*D+f] - vmin) / binWidth, B-1);
            buckets[b].push_back(i);
        }
        
        // 寻找最佳分割点
        std::vector<int> leftIdx;
        leftIdx.reserve(idx.size());
        
        for (int b = 0; b < B-1; ++b) {
            leftIdx.insert(leftIdx.end(), buckets[b].begin(), buckets[b].end());
            if (leftIdx.empty()) continue;
            
            size_t leftSize = leftIdx.size();
            size_t rightSize = idx.size() - leftSize;
            if (rightSize == 0) break;
            
            // 构造右侧索引
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