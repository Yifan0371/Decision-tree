#include "finder/HistogramEQFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

std::tuple<int, double, double>
HistogramEQFinder::findBestSplit(const std::vector<double>& X,   // 行优先特征矩阵
                                 int                        D,   // 每行特征数
                                 const std::vector<double>& y,   // 标签
                                 const std::vector<int>&    idx, // 当前节点样本索引
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const
{
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    /* ---------- 预分配可复用缓冲 ---------- */
    std::vector<int> sortedIdx(idx.size());
    std::vector<int> leftBuf, rightBuf;
    leftBuf.reserve(idx.size());
    rightBuf.reserve(idx.size());

    const int  B   = std::max(1, bins_);
    const int  per = std::max(1, static_cast<int>(idx.size()) / B);
    const size_t N = idx.size();
    const double EPS = 1e-12;

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    /* ---------- 遍历每个特征 ---------- */
    for (int f = 0; f < D; ++f) {

        /* ---- 拷贝并按特征值排序索引 ---- */
        std::copy(idx.begin(), idx.end(), sortedIdx.begin());
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return X[a * D + f] < X[b * D + f];
                  });
        if (sortedIdx.size() < 2) continue;

        /* ---- 按等频步长遍历 pivot ---- */
        for (size_t pivot = per; pivot < N; pivot += per) {
            /* 相邻值相同则跳过（避免零增益分割） */
            double vL = X[sortedIdx[pivot - 1] * D + f];
            double vR = X[sortedIdx[pivot]     * D + f];
            if (std::fabs(vR - vL) < EPS) continue;

            /* --- 构造左右子集索引（缓冲复用） --- */
            leftBuf.clear();
            rightBuf.clear();

            leftBuf.insert(leftBuf.end(),
                           sortedIdx.begin(),
                           sortedIdx.begin() + pivot);
            rightBuf.insert(rightBuf.end(),
                            sortedIdx.begin() + pivot,
                            sortedIdx.end());

            double mL = crit.nodeMetric(y, leftBuf);
            double mR = crit.nodeMetric(y, rightBuf);
            double gain = parentMetric -
                (mL * leftBuf.size() + mR * rightBuf.size()) / N;

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = 0.5 * (vL + vR);   // 阈值取中点
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}
