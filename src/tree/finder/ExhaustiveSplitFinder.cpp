// src/tree/finder/ExhaustiveSplitFinder.cpp
#include "finder/ExhaustiveSplitFinder.hpp"
#include "criterion/MSECriterion.hpp"
#include <algorithm>
#include <tuple>

#pragma GCC optimize("O3")

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int rowLength,
                                     const std::vector<double>& labels,
                                     const std::vector<int>& indices,
                                     double currentMetric,
                                     const ISplitCriterion& criterion) const {
    
    // 特化处理 MSE 情况以获得最大性能
    if (const MSECriterion* mseCrit = dynamic_cast<const MSECriterion*>(&criterion)) {
        return findBestSplitMSE(data, rowLength, labels, indices, currentMetric);
    }
    
    // 其他准则使用通用方法
    return findBestSplitGeneric(data, rowLength, labels, indices, currentMetric, criterion);
}

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplitMSE(const std::vector<double>& data,
                                        int rowLength,
                                        const std::vector<double>& labels,
                                        const std::vector<int>& indices,
                                        double currentMetric) const {
    
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestImp  = 0.0;
    const size_t nFeat = rowLength;
    const size_t n = indices.size();

    // 预分配内存，避免重复分配
    std::vector<SplitCandidate> candidates;
    candidates.reserve(n);

    for (size_t f = 0; f < nFeat; ++f) {
        candidates.clear();
        
        // 构建候选分割点，避免重复值
        for (int idx : indices) {
            candidates.push_back({data[idx * rowLength + f], idx});
        }
        
        // 排序一次，后续使用增量统计
        std::sort(candidates.begin(), candidates.end());
        
        // 移除重复值的分割点
        size_t writePos = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].value != candidates[writePos].value) {
                ++writePos;
                if (writePos != i) {
                    candidates[writePos] = candidates[i];
                }
            }
        }
        if (writePos == 0) continue; // 所有值相同
        
        // 使用增量统计避免重复计算 MSE
        // 左侧统计：sum, sumSq, count
        double leftSum = 0.0, leftSumSq = 0.0;
        size_t leftCount = 0;
        
        // 计算右侧初始统计
        double totalSum = 0.0, totalSumSq = 0.0;
        for (const auto& cand : candidates) {
            double y = labels[cand.index];
            totalSum += y;
            totalSumSq += y * y;
        }
        
        for (size_t i = 0; i < writePos; ++i) {
            double y = labels[candidates[i].index];
            leftSum += y;
            leftSumSq += y * y;
            leftCount++;
            
            // 检查是否可以分割（避免边界情况）
            size_t rightCount = n - leftCount;
            if (rightCount == 0) break;
            
            // 计算阈值
            double v1 = candidates[i].value;
            double v2 = candidates[i + 1].value;
            double thr = 0.5 * (v1 + v2);
            
            // 增量计算 MSE
            double leftMean = leftSum / leftCount;
            double leftMSE = (leftSumSq / leftCount) - leftMean * leftMean;
            
            double rightSum = totalSum - leftSum;
            double rightSumSq = totalSumSq - leftSumSq;
            double rightMean = rightSum / rightCount;
            double rightMSE = (rightSumSq / rightCount) - rightMean * rightMean;
            
            // 加权 MSE
            double weightedMSE = (leftMSE * leftCount + rightMSE * rightCount) / n;
            double imp = currentMetric - weightedMSE;
            
            if (imp > bestImp) {
                bestImp  = imp;
                bestFeat = f;
                bestThr  = thr;
            }
        }
    }
    
    return {bestFeat, bestThr, bestImp};
}

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplitGeneric(const std::vector<double>& data,
                                           int rowLength,
                                           const std::vector<double>& labels,
                                           const std::vector<int>& indices,
                                           double currentMetric,
                                           const ISplitCriterion& criterion) const {
    
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestImp  = 0.0;
    const size_t nFeat = rowLength;

    // 预分配内存
    std::vector<SplitCandidate> candidates;
    std::vector<int> leftIdx, rightIdx;
    candidates.reserve(indices.size());
    leftIdx.reserve(indices.size() / 2);
    rightIdx.reserve(indices.size() / 2);

    for (size_t f = 0; f < nFeat; ++f) {
        candidates.clear();
        
        for (int idx : indices) {
            candidates.push_back({data[idx * rowLength + f], idx});
        }
        
        std::sort(candidates.begin(), candidates.end());

        // 使用更高效的分割方式，避免 vector 的 insert/erase
        for (size_t i = 0; i + 1 < candidates.size(); ++i) {
            double v1 = candidates[i].value;
            double v2 = candidates[i+1].value;
            if (v1 == v2) continue;

            double thr = 0.5 * (v1 + v2);
            
            // 重新分配，但避免插入删除操作
            leftIdx.clear();
            rightIdx.clear();
            
            for (const auto& cand : candidates) {
                if (cand.value <= thr) {
                    leftIdx.push_back(cand.index);
                } else {
                    rightIdx.push_back(cand.index);
                }
            }
            
            if (leftIdx.empty() || rightIdx.empty()) continue;
            
            double mLeft  = criterion.nodeMetric(labels, leftIdx);
            double mRight = criterion.nodeMetric(labels, rightIdx);
            double weighted = (mLeft * leftIdx.size() +
                              mRight * rightIdx.size()) / indices.size();
            double imp = currentMetric - weighted;
            
            if (imp > bestImp) {
                bestImp  = imp;
                bestFeat = f;
                bestThr  = thr;
            }
        }
    }
    
    return {bestFeat, bestThr, bestImp};
}