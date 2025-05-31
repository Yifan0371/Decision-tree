#include "finder/QuartileSplitFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

std::tuple<int, double, double>
QuartileSplitFinder::findBestSplit(const std::vector<double>& X,   // 特征矩阵 (行优先)
                                   int                        D,   // 每行特征数
                                   const std::vector<double>& y,   // 标签
                                   const std::vector<int>&    idx, // 当前样本索引
                                   double                     parentMetric,
                                   const ISplitCriterion&     crit) const
{
    if (idx.size() < 4) return {-1, 0.0, 0.0};   // 数据太少直接返回

    /* ---- 复用缓冲区 ---- */
    std::vector<double> vals;
    vals.reserve(idx.size());

    std::vector<int> leftBuf, rightBuf;
    leftBuf.reserve(idx.size());
    rightBuf.reserve(idx.size());

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    for (int f = 0; f < D; ++f) {

        /* -------- 收集当前特征值 -------- */
        vals.clear();
        for (int i : idx)
            vals.emplace_back(X[i * D + f]);

        if (vals.size() < 4) continue;              // 再次保护

        /* -------- 一次排序直接取四分位 -------- */
        std::sort(vals.begin(), vals.end());
        const size_t N       = vals.size();
        const double q1      = vals[static_cast<size_t>(0.25 * (N - 1))];
        const double q2      = vals[static_cast<size_t>(0.50 * (N - 1))];
        const double q3      = vals[static_cast<size_t>(0.75 * (N - 1))];

        /* -------- 组织去重后的阈值数组 -------- */
        double thrList[3];               // 最多 3 个阈值
        int    thrCnt = 0;
        thrList[thrCnt++] = q1;
        if (std::fabs(q2 - q1) > 1e-12)  thrList[thrCnt++] = q2;
        if (std::fabs(q3 - q2) > 1e-12 && std::fabs(q3 - q1) > 1e-12)  thrList[thrCnt++] = q3;

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
                (mL * leftBuf.size() + mR * rightBuf.size()) / idx.size();

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = thr;
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}
