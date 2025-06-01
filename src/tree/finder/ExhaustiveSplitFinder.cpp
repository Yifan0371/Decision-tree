
#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>

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

    /* ---------- 统计父节点信息 ---------- */
    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    for (int idx : indices) {
        const double y = labels[idx];
        totalSum   += y;
        totalSumSq += y * y;               // 乘法替代 pow(y, 2)
    }
    const double parentMean = totalSum / static_cast<double>(N);
    const double parentMSE  = totalSumSq / static_cast<double>(N) - parentMean * parentMean;

    /* ---------- 预分配缓冲区 ---------- */
    std::vector<int> sortedIdx(N);

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = 0.0;
    constexpr double EPS = 1e-12;

    /* ---------- 遍历每个特征 ---------- */
    for (int f = 0; f < rowLength; ++f) {
        /* --- 拷贝当前索引并按特征值排序 --- */
        std::copy(indices.begin(), indices.end(), sortedIdx.begin());
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return data[a * rowLength + f] < data[b * rowLength + f];
                  });

        /* --- 单循环累加左子集统计量并即时评估切分 --- */
        double leftSum   = 0.0;
        double leftSumSq = 0.0;

        for (size_t i = 0; i < N - 1; ++i) {
            const int    idx = sortedIdx[i];
            const double y   = labels[idx];
            leftSum   += y;
            leftSumSq += y * y;

            /* 判断相邻样本特征值是否不同 → 是否可切分 */
            const double currentVal = data[idx * rowLength + f];
            const double nextVal    = data[sortedIdx[i + 1] * rowLength + f];

            if (__builtin_expect(currentVal + EPS < nextVal, 1)) { // 绝大多数情况下可切分
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

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeat = f;
                    bestThr  = 0.5 * (currentVal + nextVal); // 中值阈值
                }
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}
