/*  ExhaustiveSplitFinder.cpp  */
#include "finder/ExhaustiveSplitFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int                         rowLength,
                                     const std::vector<double>&  labels,
                                     const std::vector<int>&     indices,
                                     double                      /*currentMetric*/,
                                     const ISplitCriterion&      /*criterion*/) const
{
    const size_t N = indices.size();
    if (N < 2) return {-1, 0.0, 0.0};

    /* ---------- 统计父节点信息 ---------- */
    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    for (int idx : indices) {
        const double y = labels[idx];
        totalSum   += y;
        totalSumSq += y * y;
    }
    const double parentMSE = totalSumSq / N - std::pow(totalSum / N, 2);

    /* ---------- 预分配临时缓冲区 ---------- */
    std::vector<int>    sortedIdx(N);
    std::vector<double> prefixSum(N);
    std::vector<double> prefixSumSq(N);

    /* ---------- 搜索所有特征 ---------- */
    int    bestFeat  = -1;
    double bestThr   = 0.0;
    double bestGain  = 0.0;
    const  double EPS = 1e-12;

    for (int f = 0; f < rowLength; ++f) {

        /* --- 将当前 indices 拷贝到工作数组并按特征值排序（仅排序索引） --- */
        std::copy(indices.begin(), indices.end(), sortedIdx.begin());

        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return data[a * rowLength + f] < data[b * rowLength + f];
                  });

        /* --- 计算前缀和 / 平方和 --- */
        {
            double runningSum   = 0.0;
            double runningSumSq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                const double y = labels[sortedIdx[i]];
                runningSum   += y;
                runningSumSq += y * y;
                prefixSum[i]   = runningSum;
                prefixSumSq[i] = runningSumSq;
            }
        }

        /* --- 枚举 N-1 个潜在分割点 --- */
        for (size_t i = 0; i < N - 1; ++i) {
            double currentVal = data[sortedIdx[i]     * rowLength + f];
            double nextVal    = data[sortedIdx[i + 1] * rowLength + f];

            /* 特征值相同 ⇒ 无法切分 */
            if (std::fabs(currentVal - nextVal) < EPS) continue;

            const size_t leftCount  = i + 1;
            const size_t rightCount = N - leftCount;
            if (leftCount == 0 || rightCount == 0) continue;  // 应该不会触发

            /* 左子集统计量（前缀和 O(1) 获取）*/
            const double leftSum   = prefixSum[i];
            const double leftSumSq = prefixSumSq[i];
            /* 右子集统计量 */
            const double rightSum   = totalSum   - leftSum;
            const double rightSumSq = totalSumSq - leftSumSq;

            /* 计算左右子集 MSE */
            const double leftMSE  = leftSumSq  / leftCount  - std::pow(leftSum  / leftCount,  2);
            const double rightMSE = rightSumSq / rightCount - std::pow(rightSum / rightCount, 2);

            /* 信息增益 = 父节点 MSE - 加权子节点 MSE */
            const double weightedMSE = (leftMSE * leftCount + rightMSE * rightCount) / N;
            const double gain        = parentMSE - weightedMSE;

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = 0.5 * (currentVal + nextVal);  // 取中值作为阈值
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}
