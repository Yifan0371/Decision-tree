#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <tuple>
#include <limits>

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int rowLength,
                                     const std::vector<double>& labels,
                                     const std::vector<int>& indices,
                                     double currentMetric,
                                     const ISplitCriterion& criterion) const {
    
    if (indices.size() < 2) {
        return {-1, 0.0, 0.0};
    }
    
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = 0.0;
    size_t N = indices.size();
    
    // 预计算父节点的总和与平方和
    double totalSum = 0.0;
    double totalSumSq = 0.0;
    for (int idx : indices) {
        double y = labels[idx];
        totalSum += y;
        totalSumSq += y * y;
    }
    
    // 父节点的MSE（应该等于currentMetric）
    double parentMSE = totalSumSq / N - (totalSum / N) * (totalSum / N);
    
    // 遍历所有特征
    for (int f = 0; f < rowLength; ++f) {
        // 创建 (特征值, 索引) 对并排序
        std::vector<std::pair<double, int>> sortedPairs;
        sortedPairs.reserve(indices.size());
        
        for (int idx : indices) {
            sortedPairs.emplace_back(data[idx * rowLength + f], idx);
        }
        
        std::sort(sortedPairs.begin(), sortedPairs.end());
        
        // 使用前缀和技巧进行增量计算
        double leftSum = 0.0;
        double leftSumSq = 0.0;
        size_t leftCount = 0;
        
        // 遍历所有可能的分割点
        for (size_t i = 0; i < sortedPairs.size() - 1; ++i) {
            int idx = sortedPairs[i].second;
            double y = labels[idx];
            
            // 将当前样本加入左子集
            leftSum += y;
            leftSumSq += y * y;
            leftCount++;
            
            // 检查是否可以在此处分割（特征值不同）
            double currentVal = sortedPairs[i].first;
            double nextVal = sortedPairs[i + 1].first;
            
            if (std::abs(currentVal - nextVal) < 1e-12) {
                continue; // 特征值相同，跳过
            }
            
            // 计算右子集的统计量
            size_t rightCount = N - leftCount;
            double rightSum = totalSum - leftSum;
            double rightSumSq = totalSumSq - leftSumSq;
            
            if (leftCount == 0 || rightCount == 0) {
                continue;
            }
            
            // 使用公式计算MSE：MSE = E[y²] - (E[y])²
            double leftMSE = leftSumSq / leftCount - (leftSum / leftCount) * (leftSum / leftCount);
            double rightMSE = rightSumSq / rightCount - (rightSum / rightCount) * (rightSum / rightCount);
            
            // 计算加权平均MSE
            double weightedMSE = (leftMSE * leftCount + rightMSE * rightCount) / N;
            
            // 计算信息增益
            double gain = parentMSE - weightedMSE;
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr = 0.5 * (currentVal + nextVal);
            }
        }
    }
    
    return {bestFeat, bestThr, bestGain};
}