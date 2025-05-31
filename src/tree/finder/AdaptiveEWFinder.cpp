#include "finder/AdaptiveEWFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

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
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    /* 可复用的缓冲区 */
    std::vector<double>           values;   values.reserve(idx.size());
    std::vector<std::vector<int>> buckets;
    std::vector<int>              leftBuf, rightBuf;
    leftBuf.reserve(idx.size());
    rightBuf.reserve(idx.size());

    const size_t N   = idx.size();
    const double EPS = 1e-12;

    for (int f = 0; f < rowLen; ++f) {

        /* 1. 收集特征值 */
        values.clear();
        for (int i : idx) values.emplace_back(data[i * rowLen + f]);

        /* 2. 计算自适应箱数 */
        const int B = calculateOptimalBins(values);
        if (B < 2) continue;

        /* 3. 计算 min / max */
        const auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
        const double vMin = *vMinIt;
        const double vMax = *vMaxIt;
        if (vMax - vMin < EPS) continue;

        const double binW = (vMax - vMin) / B;

        /* 4. 分桶 */
        buckets.assign(B, {});
        for (int i : idx) {
            int b = static_cast<int>((data[i * rowLen + f] - vMin) / binW);
            if (b == B) b--;
            buckets[b].push_back(i);
        }

        /* 5. 线性扫描分割点 */
        leftBuf.clear();
        for (int b = 0; b < B - 1; ++b) {
            leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
            if (leftBuf.empty()) continue;

            const size_t leftN  = leftBuf.size();
            const size_t rightN = N - leftN;
            if (rightN == 0) break;

            rightBuf.clear();
            for (int k = b + 1; k < B; ++k)
                rightBuf.insert(rightBuf.end(), buckets[k].begin(), buckets[k].end());

            double mL = criterion.nodeMetric(labels, leftBuf);
            double mR = criterion.nodeMetric(labels, rightBuf);
            double gain = parentMetric -
                (mL * leftN + mR * rightN) / static_cast<double>(N);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = vMin + binW * (b + 1);
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}
