#include "finder/HistogramEWFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

std::tuple<int, double, double>
HistogramEWFinder::findBestSplit(const std::vector<double>& X,   // 行优先特征矩阵
                                 int                        D,   // 每行特征数
                                 const std::vector<double>& y,   // 标签
                                 const std::vector<int>&    idx, // 当前节点样本索引
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const
{
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    /* ---------- 预分配并复用直方图缓冲 ---------- */
    std::vector<int>    histCnt(bins_);
    std::vector<double> histSum(bins_);
    std::vector<double> histSumSq(bins_);

    std::vector<int>    prefixCnt(bins_);
    std::vector<double> prefixSum(bins_);
    std::vector<double> prefixSumSq(bins_);

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;
    const size_t N   = idx.size();

    for (int f = 0; f < D; ++f) {

        /* ---- 求该特征的 (min,max) ---- */
        double vMin =  std::numeric_limits<double>::infinity();
        double vMax = -vMin;
        for (int i : idx) {
            double v = X[i * D + f];
            vMin = std::min(vMin, v);
            vMax = std::max(vMax, v);
        }
        if (std::fabs(vMax - vMin) < EPS) continue;   // 无法切分

        const double binW = (vMax - vMin) / bins_;

        /* ---- 清空直方图数组 ---- */
        std::fill(histCnt.begin(),   histCnt.end(),   0);
        std::fill(histSum.begin(),   histSum.end(),   0.0);
        std::fill(histSumSq.begin(), histSumSq.end(), 0.0);

        /* ---- 一次扫描样本，填充直方图 ---- */
        for (int i : idx) {
            const double v = X[i * D + f];
            int   b = static_cast<int>((v - vMin) / binW);
            if (b == bins_) b--;          // 把边界点归到最后一 bin
            const double lbl = y[i];

            histCnt  [b] += 1;
            histSum  [b] += lbl;
            histSumSq[b] += lbl * lbl;
        }

        /* ---- 前缀累加 (左子集统计) ---- */
        prefixCnt[0]   = histCnt[0];
        prefixSum[0]   = histSum[0];
        prefixSumSq[0] = histSumSq[0];
        for (int b = 1; b < bins_; ++b) {
            prefixCnt [b] = prefixCnt [b-1] + histCnt [b];
            prefixSum [b] = prefixSum [b-1] + histSum [b];
            prefixSumSq[b] = prefixSumSq[b-1] + histSumSq[b];
        }

        /* ---- 遍历 bins_-1 个候选切分点 ---- */
        for (int b = 0; b < bins_ - 1; ++b) {
            const int    leftCnt  = prefixCnt[b];
            const int    rightCnt = static_cast<int>(N) - leftCnt;
            if (leftCnt == 0 || rightCnt == 0) continue;

            const double leftSum   = prefixSum[b];
            const double leftSumSq = prefixSumSq[b];
            const double rightSum   = prefixSum[bins_-1] - leftSum;
            const double rightSumSq = prefixSumSq[bins_-1] - leftSumSq;

            const double leftMSE  = leftSumSq  / leftCnt  - std::pow(leftSum  / leftCnt , 2);
            const double rightMSE = rightSumSq / rightCnt - std::pow(rightSum / rightCnt, 2);
            const double gain = parentMetric -
                (leftMSE * leftCnt + rightMSE * rightCnt) / N;

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = vMin + (b + 0.5) * binW;   // bin 中点
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}
