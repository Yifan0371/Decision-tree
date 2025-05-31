#include "finder/AdaptiveEQFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

/*=== 工具函数 =============================================================*/
static double coeffOfVariation(const std::vector<double>& v)
{
    if (v.size() <= 1) return 0.0;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= v.size();
    return std::sqrt(var) / (std::fabs(mean) + 1e-12);
}

/*=== AdaptiveEQFinder 私有辅助 ===========================================*/
std::pair<int,int>
AdaptiveEQFinder::calculateOptimalFrequencyParams(const std::vector<double>& v) const
{
    const int n  = static_cast<int>(v.size());
    const double cv = coeffOfVariation(v);

    int bins = (cv < variabilityThreshold_)
             ? std::max(4, std::min(16, static_cast<int>(std::sqrt(n) / 2)))
             : std::max(8, std::min(maxBins_, static_cast<int>(std::sqrt(n))));
    bins = std::clamp(bins, 2, n / std::max(1, minSamplesPerBin_));  // 至少 2 盒

    int perBin = std::max(minSamplesPerBin_, n / bins);
    return {bins, perBin};
}

/*=== findBestSplit =======================================================*/
std::tuple<int,double,double>
AdaptiveEQFinder::findBestSplit(const std::vector<double>& data,
                                int                       rowLen,
                                const std::vector<double>&labels,
                                const std::vector<int>&   idx,
                                double                    parentMetric,
                                const ISplitCriterion&    criterion) const
{
    const size_t N = idx.size();
    if (N < static_cast<size_t>(2 * minSamplesPerBin_)) return {-1, 0.0, 0.0};

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    std::vector<double> values; values.reserve(N);
    std::vector<int>    sortedIdx(N);
    std::vector<int>    leftBuf, rightBuf;
    leftBuf.reserve(N);
    rightBuf.reserve(N);

    const double EPS = 1e-12;

    for (int f = 0; f < rowLen; ++f) {

        /* 1. 收集当前特征值 */
        values.clear();
        for (int i : idx) values.emplace_back(data[i * rowLen + f]);

        /* 2. 计算自适应等频参数 */
        const auto [bins, perBin] = calculateOptimalFrequencyParams(values);
        if (N < static_cast<size_t>(2 * perBin)) continue;

        /* 3. 索引排序（避免 pair 复制） */
        std::copy(idx.begin(), idx.end(), sortedIdx.begin());
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return data[a * rowLen + f] < data[b * rowLen + f];
                  });

        /* 4. 枚举等频分割 */
        for (size_t pivot = perBin;
             pivot <= N - perBin;
             pivot += perBin)
        {
            const double vL = data[sortedIdx[pivot - 1] * rowLen + f];
            const double vR = data[sortedIdx[pivot]     * rowLen + f];
            if (std::fabs(vR - vL) < EPS) continue;      // 相同值无效

            leftBuf.assign(sortedIdx.begin(),          sortedIdx.begin() + pivot);
            rightBuf.assign(sortedIdx.begin() + pivot, sortedIdx.end());

            if (leftBuf.size()  < static_cast<size_t>(minSamplesPerBin_) ||
                rightBuf.size() < static_cast<size_t>(minSamplesPerBin_))
                continue;

            const double mL = criterion.nodeMetric(labels, leftBuf);
            const double mR = criterion.nodeMetric(labels, rightBuf);
            const double gain = parentMetric -
                (mL * leftBuf.size() + mR * rightBuf.size()) / static_cast<double>(N);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeat = f;
                bestThr  = 0.5 * (vL + vR);
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}
