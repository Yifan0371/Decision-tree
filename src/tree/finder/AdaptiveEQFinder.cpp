#include "finder/AdaptiveEQFinder.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

double AdaptiveEQFinder::calculateVariability(const std::vector<double>& values) const {
    if (values.size() <= 1) return 0.0;
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    for (double v : values) {
        variance += (v - mean) * (v - mean);
    }
    variance /= values.size();
    
    // 返回变异系数（标准差/均值）
    return std::sqrt(variance) / (std::abs(mean) + 1e-12);
}

std::pair<int, int> AdaptiveEQFinder::calculateOptimalFrequencyParams(
    const std::vector<double>& values) const
{
    int n = static_cast<int>(values.size());
    
    // 计算数据的变异性
    double variability = calculateVariability(values);
    
    // 根据变异性调整策略
    int optimalBins;
    int samplesPerBin;
    
    if (variability < variabilityThreshold_) {
        // 低变异性：使用较少的箱子，每箱更多样本
        optimalBins = std::max(4, std::min(16, static_cast<int>(std::sqrt(n) / 2)));
        samplesPerBin = std::max(minSamplesPerBin_ * 2, n / optimalBins);
    } else {
        // 高变异性：使用更多箱子，允许较少样本
        optimalBins = std::max(8, std::min(maxBins_, static_cast<int>(std::sqrt(n))));
        samplesPerBin = std::max(minSamplesPerBin_, n / optimalBins);
    }
    
    // 确保至少有2个箱子
    optimalBins = std::max(2, optimalBins);
    
    // 根据最小样本数约束调整箱数
    int maxPossibleBins = n / minSamplesPerBin_;
    optimalBins = std::min(optimalBins, maxPossibleBins);
    
    return {optimalBins, samplesPerBin};
}

std::tuple<int,double,double> AdaptiveEQFinder::findBestSplit(
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
        
        for (int i : idx) {
            values.push_back(X[i*D+f]);
        }
        
        // 计算自适应参数
        auto [optimalBins, samplesPerBin] = calculateOptimalFrequencyParams(values);
        
        // 创建排序的(值,索引)对
        std::vector<std::pair<double, int>> pairs;
        pairs.reserve(idx.size());
        for (int i : idx) pairs.emplace_back(X[i*D+f], i);
        std::sort(pairs.begin(), pairs.end());
        
        if (static_cast<int>(pairs.size()) < 2 * minSamplesPerBin_) continue;
        
        // 动态调整每箱样本数，确保能产生有意义的分割
        int actualSamplesPerBin = std::max(minSamplesPerBin_, 
                                          static_cast<int>(pairs.size()) / optimalBins);
        
        // 尝试不同的分割点
        for (size_t pivot = static_cast<size_t>(actualSamplesPerBin); 
             pivot < pairs.size() - static_cast<size_t>(actualSamplesPerBin); 
             pivot += static_cast<size_t>(actualSamplesPerBin)) {
                
            std::vector<int> leftIdx, rightIdx;
            leftIdx.reserve(pivot);
            rightIdx.reserve(pairs.size() - pivot);
            
            for (size_t i = 0; i < pivot; ++i) 
                leftIdx.push_back(pairs[i].second);
            for (size_t i = pivot; i < pairs.size(); ++i) 
                rightIdx.push_back(pairs[i].second);
                
            if (static_cast<int>(leftIdx.size()) < minSamplesPerBin_ || 
                static_cast<int>(rightIdx.size()) < minSamplesPerBin_) continue;
            
            double mL = crit.nodeMetric(y, leftIdx);
            double mR = crit.nodeMetric(y, rightIdx);
            double gain = parentMetric - (mL*leftIdx.size() + mR*rightIdx.size()) / idx.size();
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr = 0.5 * (pairs[pivot-1].first + pairs[pivot].first);
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}