// HistogramEQFinder.cpp
#include "finder/HistogramEQFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>  // 新增 OpenMP 头文件

std::tuple<int, double, double>
HistogramEQFinder::findBestSplit(const std::vector<double>& X,   // 行优先特征矩阵
                                 int                        D,   // 每行特征数
                                 const std::vector<double>& y,   // 标签
                                 const std::vector<int>&    idx, // 当前节点样本索引
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const
{
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    /* ---------- 预分配可复用缓冲 ---------- */
    std::vector<int> sortedIdx(idx.size());
    std::vector<int> leftBuf, rightBuf;
    leftBuf.reserve(N);
    rightBuf.reserve(N);

    const int B   = std::max(1, bins_);
    const int per = std::max(1, static_cast<int>(N) / B);
    const double EPS = 1e-12;

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    /* ---------- 并行遍历每个特征 ---------- */
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < D; ++f) {
        // 每个线程维护自己的局部最优
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // 1. 拷贝并按特征值排序索引
        std::vector<int> localSorted = idx;  // 每个线程独立的排序缓冲
        std::sort(localSorted.begin(), localSorted.end(),
                  [&](int a, int b) {
                      return X[a * D + f] < X[b * D + f];
                  });
        if (localSorted.size() < 2) continue;

        // 2. 按等频步长遍历 pivot
        for (size_t pivot = per; pivot < N; pivot += per) {
            double vL = X[localSorted[pivot - 1] * D + f];
            double vR = X[localSorted[pivot]     * D + f];
            if (std::fabs(vR - vL) < EPS) continue;  // 相邻值相同跳过

            // 构造左右子集索引（每个线程独立缓冲）
            std::vector<int> localLeft, localRight;
            localLeft.reserve(pivot);
            localRight.reserve(N - pivot);

            localLeft.insert(localLeft.end(),
                             localSorted.begin(),
                             localSorted.begin() + pivot);
            localRight.insert(localRight.end(),
                              localSorted.begin() + pivot,
                              localSorted.end());

            // 计算左右子节点的度量值（criterion 内部可能也并行）
            double mL = crit.nodeMetric(y, localLeft);
            double mR = crit.nodeMetric(y, localRight);
            double gain = parentMetric -
                          (mL * localLeft.size() + mR * localRight.size()) / static_cast<double>(N);

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = 0.5 * (vL + vR);  // 阈值取中点
            }
        }

        // 用 critical 区块更新全局最优
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeat = f;
                bestThr  = localBestThr;
            }
        }
    } // 并行 for 结束

    return {bestFeat, bestThr, bestGain};
}
