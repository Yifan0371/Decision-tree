// AdaptiveEQFinder.cpp
#include "finder/AdaptiveEQFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>   // 新增 OpenMP 头文件

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
    if (N < static_cast<size_t>(2 * minSamplesPerBin_))
        return {-1, 0.0, 0.0};

    // 全局最优初始化
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // 并行遍历每个特征 f
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < rowLen; ++f) {
        // 每个线程维护自己的局部最优
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // 1. 并行收集当前特征值到局部 vectors
        std::vector<double> values;
        values.reserve(N);
        for (int i : idx) {
            values.push_back(data[i * rowLen + f]);
        }

        // 2. 计算自适应等频参数（计算 perBin, bins）
        const auto [bins, perBin] = calculateOptimalFrequencyParams(values);
        if (N < static_cast<size_t>(2 * perBin)) continue;  // 样本太少，跳过此特征

        // 3. 对索引进行排序，得到 sortedIdx
        std::vector<int> sortedIdx = idx;  // 直接拷贝
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return data[a * rowLen + f] < data[b * rowLen + f];
                  });

        // 4. 枚举等频分割点
        for (size_t pivot = perBin; pivot <= N - perBin; pivot += perBin) {
            double vL = data[sortedIdx[pivot - 1] * rowLen + f];
            double vR = data[sortedIdx[pivot]     * rowLen + f];
            if (std::fabs(vR - vL) < EPS) 
                continue;  // 相同值无效

            // 分配到左右子节点索引缓冲区
            std::vector<int> leftBuf, rightBuf;
            leftBuf.reserve(pivot);
            rightBuf.reserve(N - pivot);
            leftBuf.assign(sortedIdx.begin(),          sortedIdx.begin() + pivot);
            rightBuf.assign(sortedIdx.begin() + pivot, sortedIdx.end());

            // 保证每个子节点样本数 >= minSamplesPerBin_
            if (leftBuf.size() < static_cast<size_t>(minSamplesPerBin_) ||
                rightBuf.size() < static_cast<size_t>(minSamplesPerBin_))
            {
                continue;
            }

            // 计算左右子节点的度量值（此处 criterion.nodeMetric 内部已可并行）
            double mL = criterion.nodeMetric(labels, leftBuf);
            double mR = criterion.nodeMetric(labels, rightBuf);
            double gain = parentMetric -
                          (mL * leftBuf.size() + mR * rightBuf.size())
                          / static_cast<double>(N);

            // 更新局部最优
            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = 0.5 * (vL + vR);
            }
        }

        // 更新到全局最优时要加锁
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
