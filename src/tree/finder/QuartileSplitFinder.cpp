// QuartileSplitFinder.cpp
#include "finder/QuartileSplitFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>   // 新增 OpenMP 头文件

std::tuple<int, double, double>
QuartileSplitFinder::findBestSplit(const std::vector<double>& X,   // 特征矩阵 (行优先)
                                   int                        D,   // 每行特征数
                                   const std::vector<double>& y,   // 标签
                                   const std::vector<int>&    idx, // 当前样本索引
                                   double                     parentMetric,
                                   const ISplitCriterion&     crit) const
{
    if (idx.size() < 4) return {-1, 0.0, 0.0};   // 数据太少直接返回

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const size_t N = idx.size();
    const double EPS = 1e-12;

    /* 并行遍历每个特征 f */
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < D; ++f) {
        // 每个线程维护自己的局部最优
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // ---- 当前线程独有的缓冲区 ----
        std::vector<double> vals;
        vals.reserve(N);

        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(N);
        rightBuf.reserve(N);

        /* -------- 收集当前特征值 -------- */
        for (int i : idx) {
            vals.emplace_back(X[i * D + f]);
        }
        if (vals.size() < 4) {
            continue;  // 再次保护
        }

        /* -------- 一次排序直接取四分位 -------- */
        std::sort(vals.begin(), vals.end());
        const size_t nVals    = vals.size();
        const double q1       = vals[static_cast<size_t>(0.25 * (nVals - 1))];
        const double q2       = vals[static_cast<size_t>(0.50 * (nVals - 1))];
        const double q3       = vals[static_cast<size_t>(0.75 * (nVals - 1))];

        /* -------- 组织去重后的阈值数组 -------- */
        double thrList[3];
        int    thrCnt = 0;
        thrList[thrCnt++] = q1;
        if (std::fabs(q2 - q1) > EPS)  thrList[thrCnt++] = q2;
        if (std::fabs(q3 - q2) > EPS && std::fabs(q3 - q1) > EPS)  thrList[thrCnt++] = q3;

        /* -------- 依次评估每个阈值 -------- */
        for (int t = 0; t < thrCnt; ++t) {
            const double thr = thrList[t];

            leftBuf.clear();
            rightBuf.clear();
            for (int i : idx) {
                if (X[i * D + f] <= thr)
                    leftBuf.emplace_back(i);
                else
                    rightBuf.emplace_back(i);
            }
            if (leftBuf.empty() || rightBuf.empty()) continue;

            const double mL = crit.nodeMetric(y, leftBuf);
            const double mR = crit.nodeMetric(y, rightBuf);
            const double gain = parentMetric -
                                (mL * leftBuf.size() + mR * rightBuf.size()) / static_cast<double>(N);

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = thr;
            }
        }

        /* 更新全局最优（加锁） */
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
