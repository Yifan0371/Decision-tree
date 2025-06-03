// AdaptiveEWFinder.cpp
#include "finder/AdaptiveEWFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>   // 新增 OpenMP 头文件

/*=== 工具函数 =============================================================*/
static double interQuartileRange(std::vector<double>& v)
{
    if (v.size() < 4) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return v[3 * n / 4] - v[n / 4];
}

/*=== AdaptiveEWFinder 成员函数 ===========================================*/

int AdaptiveEWFinder::calculateOptimalBins(const std::vector<double>& values) const
{
    const int n = static_cast<int>(values.size());
    if (n <= 1) return 1;

    int bins = minBins_;

    if (rule_ == "sturges") {
        bins = static_cast<int>(std::ceil(std::log2(n))) + 1;
    } else if (rule_ == "rice") {
        bins = static_cast<int>(std::ceil(2.0 * std::cbrt(n)));
    } else if (rule_ == "sqrt") {
        bins = static_cast<int>(std::ceil(std::sqrt(n)));
    } else if (rule_ == "freedman_diaconis") {
        std::vector<double> tmp(values);
        const double iqr = interQuartileRange(tmp);
        if (iqr > 0.0) {
            const auto [mn, mx] = std::minmax_element(tmp.begin(), tmp.end());
            const double h   = 2.0 * iqr / std::cbrt(n);
            bins = static_cast<int>(std::ceil((*mx - *mn) / h));
        }
    }
    return std::clamp(bins, minBins_, maxBins_);
}

/* 备用接口：当前实现未直接调用，可留作外部使用 */
double AdaptiveEWFinder::calculateIQR(std::vector<double> values) const
{
    return interQuartileRange(values);
}

/*=== findBestSplit =======================================================*/
std::tuple<int, double, double>
AdaptiveEWFinder::findBestSplit(const std::vector<double>& data,
                                int                       rowLen,
                                const std::vector<double>&labels,
                                const std::vector<int>&   idx,
                                double                    parentMetric,
                                const ISplitCriterion&    criterion) const
{
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    // 全局最优初始化
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // 并行遍历每个特征 f
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < rowLen; ++f) {
        // 每个线程维护自己针对当前特征 f 的局部最优
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // 1. 收集当前特征值
        std::vector<double> values;
        values.reserve(N);
        for (int i : idx) {
            values.emplace_back(data[i * rowLen + f]);
        }

        // 2. 计算自适应箱数
        const int B = calculateOptimalBins(values);
        if (B < 2) continue;  // 箱数不足，跳过此特征

        // 3. 计算最小值和最大值
        const auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
        const double vMin = *vMinIt;
        const double vMax = *vMaxIt;
        if (vMax - vMin < EPS) continue;  // 无效特征，跳过

        const double binW = (vMax - vMin) / B;

        // 4. 分桶：每个线程维护自己的 buckets
        std::vector<std::vector<int>> buckets(B);
        for (int i : idx) {
            int b = static_cast<int>((data[i * rowLen + f] - vMin) / binW);
            if (b == B) b = B - 1;  // 边界情况
            buckets[b].push_back(i);
        }

        // 5. 线性扫描分割点
        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(N);
        rightBuf.reserve(N);

        for (int b = 0; b < B - 1; ++b) {
            // 在桶 b 边界前，把该桶所有样本添加到 leftBuf
            leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
            if (leftBuf.empty()) continue;

            size_t leftN  = leftBuf.size();
            size_t rightN = N - leftN;
            if (rightN == 0) break;

            // 准备 rightBuf：把剩余桶中的索引都收集到 rightBuf
            rightBuf.clear();
            for (int k = b + 1; k < B; ++k) {
                rightBuf.insert(rightBuf.end(), buckets[k].begin(), buckets[k].end());
            }

            // 计算左右子节点的度量值（criterion 可能内部也并行）
            double mL = criterion.nodeMetric(labels, leftBuf);
            double mR = criterion.nodeMetric(labels, rightBuf);

            double gain = parentMetric -
                          (mL * leftN + mR * rightN) / static_cast<double>(N);

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = vMin + binW * (b + 1);
            }
        }

        // 将当前特征的局部最优与全局最优比较，必要时更新全局
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
