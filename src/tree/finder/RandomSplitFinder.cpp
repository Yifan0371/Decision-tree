#include "finder/RandomSplitFinder.hpp"
#include <limits>
#include <random>
#include <vector>
#include <cmath>

/* ------------------------------------------------------------------ */
/* 新版实现：                                                          */
/* - 单线程、无依赖编译器矢量化                                        */
/* - 复用 L/R 缓冲，去掉 unordered_set                                 */
/* ------------------------------------------------------------------ */
std::tuple<int, double, double>
RandomSplitFinder::findBestSplit(const std::vector<double>& X,   // 特征矩阵 (按行)
                                 int                          D, // 每行特征数
                                 const std::vector<double>&  y,  // 标签
                                 const std::vector<int>&     idx,// 当前样本索引
                                 double                      parentMetric,
                                 const ISplitCriterion&      crit) const
{
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    int    bestFeat  = -1;
    double bestThr   = 0.0;
    double bestGain  = -std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    /* ---------- 预分配左右缓冲区，反复复用 ---------- */
    std::vector<int> leftBuf, rightBuf;
    leftBuf.reserve(idx.size());
    rightBuf.reserve(idx.size());

    /* ---------- 穷举每个特征 ---------- */
    for (int f = 0; f < D; ++f) {

        /* --- 计算该特征的最小/最大值 --- */
        double vMin =  std::numeric_limits<double>::infinity();
        double vMax = -vMin;
        for (int i : idx) {
            double v = X[i * D + f];
            vMin = std::min(vMin, v);
            vMax = std::max(vMax, v);
        }
        if (std::fabs(vMax - vMin) < 1e-12) continue;  // 全部相等，跳过

        /* --- 对该特征尝试 k_ 个随机阈值 --- */
        for (int r = 0; r < k_; ++r) {
            double thr = vMin + uni01(gen_) * (vMax - vMin);

            leftBuf.clear();
            rightBuf.clear();
            for (int i : idx) {
                if (X[i * D + f] <= thr)
                    leftBuf.emplace_back(i);
                else
                    rightBuf.emplace_back(i);
            }
            if (leftBuf.empty() || rightBuf.empty()) continue;

            double mL = crit.nodeMetric(y, leftBuf);
            double mR = crit.nodeMetric(y, rightBuf);
            double gain = parentMetric -
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
