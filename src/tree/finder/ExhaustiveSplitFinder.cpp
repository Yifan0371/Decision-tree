// src/tree/finder/ExhaustiveSplitFinder.cpp - OpenMP并行版本
#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int                       rowLength,
                                     const std::vector<double>& labels,
                                     const std::vector<int>&    indices,
                                     double /*currentMetric*/,
                                     const ISplitCriterion&     /*criterion*/) const
{
    const size_t N = indices.size();
    if (N < 2) return {-1, 0.0, 0.0};

    /* ---------- 并行计算父节点统计信息 ---------- */
    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    
    // 使用OpenMP并行reduction计算父节点统计
    #pragma omp parallel for reduction(+:totalSum,totalSumSq) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const double y = labels[indices[i]];
        totalSum   += y;
        totalSumSq += y * y;
    }
    
    const double parentMean = totalSum / static_cast<double>(N);
    const double parentMSE  = totalSumSq / static_cast<double>(N) - parentMean * parentMean;

    /* ---------- 并行特征搜索 ---------- */
    int    globalBestFeat = -1;
    double globalBestThr  = 0.0;
    double globalBestGain = 0.0;
    constexpr double EPS = 1e-12;

    // 使用OpenMP并行遍历特征，每个线程处理部分特征
    #pragma omp parallel
    {
        // 线程局部变量
        int    localBestFeat = -1;
        double localBestThr  = 0.0;
        double localBestGain = 0.0;
        
        // 线程局部缓冲区（避免重复分配）
        std::vector<int> localSortedIdx(N);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < rowLength; ++f) {
            /* --- 拷贝当前索引并按特征值排序 --- */
            std::copy(indices.begin(), indices.end(), localSortedIdx.begin());
            std::sort(localSortedIdx.begin(), localSortedIdx.end(),
                      [&](int a, int b) {
                          return data[a * rowLength + f] < data[b * rowLength + f];
                      });

            /* --- 单循环累加左子集统计量并即时评估切分 --- */
            double leftSum   = 0.0;
            double leftSumSq = 0.0;

            for (size_t i = 0; i < N - 1; ++i) {
                const int    idx = localSortedIdx[i];
                const double y   = labels[idx];
                leftSum   += y;
                leftSumSq += y * y;

                /* 判断相邻样本特征值是否不同 → 是否可切分 */
                const double currentVal = data[idx * rowLength + f];
                const double nextVal    = data[localSortedIdx[i + 1] * rowLength + f];

                if (currentVal + EPS < nextVal) {
                    const size_t leftCnt  = i + 1;
                    const size_t rightCnt = N - leftCnt;

                    /* 右子集统计量可由总量减左子集得到 */
                    const double rightSum   = totalSum   - leftSum;
                    const double rightSumSq = totalSumSq - leftSumSq;

                    /* 计算左右子集方差 */
                    const double leftMean  = leftSum  / static_cast<double>(leftCnt);
                    const double rightMean = rightSum / static_cast<double>(rightCnt);

                    const double leftMSE  = leftSumSq  / static_cast<double>(leftCnt)  - leftMean  * leftMean;
                    const double rightMSE = rightSumSq / static_cast<double>(rightCnt) - rightMean * rightMean;

                    /* 信息增益 */
                    const double gain = parentMSE -
                                         (leftMSE * static_cast<double>(leftCnt) +
                                          rightMSE * static_cast<double>(rightCnt)) / static_cast<double>(N);

                    if (gain > localBestGain) {
                        localBestGain = gain;
                        localBestFeat = f;
                        localBestThr  = 0.5 * (currentVal + nextVal);
                    }
                }
            }
        }
        
        /* --- 线程间归约：更新全局最佳结果 --- */
        #pragma omp critical
        {
            if (localBestGain > globalBestGain) {
                globalBestGain = localBestGain;
                globalBestFeat = localBestFeat;
                globalBestThr  = localBestThr;
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}