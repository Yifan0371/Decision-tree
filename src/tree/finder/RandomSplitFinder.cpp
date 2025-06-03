// src/tree/finder/RandomSplitFinder.cpp
#include "finder/RandomSplitFinder.hpp"
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double>
RandomSplitFinder::findBestSplit(const std::vector<double>& X,
                                 int                          D,
                                 const std::vector<double>&   y,
                                 const std::vector<int>&      idx,
                                 double                       parentMetric,
                                 const ISplitCriterion&       crit) const
{
    const int nIdx = static_cast<int>(idx.size());
    if (nIdx < 2) {
        return {-1, 0.0, 0.0};
    }

    // 自适应阈值：当节点样本数较小（如 < 1000）时，用串行方式
    const int PARALLEL_THRESHOLD = 1000;
    bool useParallel = (nIdx >= PARALLEL_THRESHOLD);

    int    globalBestFeat  = -1;
    double globalBestThr   = 0.0;
    double globalBestGain  = -std::numeric_limits<double>::infinity();

    // 用于并行时，给每个线程准备一个种子
    int maxThreads = 1;
#ifdef _OPENMP
    maxThreads = omp_get_max_threads();
#endif
    std::vector<uint32_t> threadSeeds(maxThreads);
    {
        // 先序列化地为每个线程生成不同 seed
        std::mt19937 seedGen(gen_());
        std::uniform_int_distribution<uint32_t> seedDist(0, 0xFFFFFFFF);
        for (int t = 0; t < maxThreads; ++t) {
            threadSeeds[t] = seedDist(seedGen);
        }
    }

    // 并行时，每个线程保存自己找到的局部最优 (feat, thr, gain)
    std::vector<int>    bestFeatPerThread(maxThreads, -1);
    std::vector<double> bestThrPerThread(maxThreads, 0.0);
    std::vector<double> bestGainPerThread(maxThreads,
                                          -std::numeric_limits<double>::infinity());

    // 包装一个函数：在单线程或并行内部调用的，计算某个特征 f 上的最优随机切分
    auto processFeature = [&](int f, int tid) {
        // 1) 提取该特征在所有节点样本中的值 (values)，以及对应的 y 标签 (labels_f)
        static thread_local std::vector<std::pair<double,double>> vals; 
        vals.clear();
        vals.reserve(nIdx);
        for (int i = 0; i < nIdx; ++i) {
            int sampleIdx = idx[i];
            double xv = X[sampleIdx * D + f];
            vals.emplace_back(xv, y[sampleIdx]);
        }

        // 2) 按特征值排序
        std::sort(vals.begin(), vals.end(),
                  [](auto &a, auto &b) { return a.first < b.first; });

        // 3) 构造前缀和数组：prefixSum[i] = sum_{j< i} vals[j].second
        //    prefixSumSq[i] = sum_{j< i} (vals[j].second)^2
        static thread_local std::vector<double> prefixSum, prefixSumSq, sortedX;
        prefixSum.resize(nIdx + 1);
        prefixSumSq.resize(nIdx + 1);
        sortedX.resize(nIdx);
        prefixSum[0]   = 0.0;
        prefixSumSq[0] = 0.0;
        for (int i = 0; i < nIdx; ++i) {
            sortedX[i] = vals[i].first;
            double yi  = vals[i].second;
            prefixSum[i+1]   = prefixSum[i]   + yi;
            prefixSumSq[i+1] = prefixSumSq[i] + yi * yi;
        }

        // 4) 根据 parentMetric（父节点的 MSE），进行 k_ 次随机阈值尝试
        std::mt19937 localGen(threadSeeds[tid]);
        std::uniform_real_distribution<double> uni01(0.0, 1.0);

        double vMin = sortedX.front();
        double vMax = sortedX.back();
        if (vMax - vMin < 1e-12) {
            return; // 该特征只有单一值，无法切分
        }

        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        for (int r = 0; r < k_; ++r) {
            double thr = vMin + uni01(localGen) * (vMax - vMin);
            // 二分查找阈值在 sortedX 中的位置 pos（第一个 > thr）
            int pos = int(std::upper_bound(sortedX.begin(), sortedX.end(), thr) - sortedX.begin());
            if (pos == 0 || pos == nIdx) {
                // 划分导致一侧为空，跳过
                continue;
            }
            // 左子集 [0, pos), 右子集 [pos, nIdx)
            double sumL   = prefixSum[pos];
            double sumSqL = prefixSumSq[pos];
            double nL     = static_cast<double>(pos);
            double mL     = sumL / nL;
            double varL   = (sumSqL / nL) - (mL * mL);

            double sumTotal   = prefixSum[nIdx];
            double sumSqTotal = prefixSumSq[nIdx];
            double sumR       = sumTotal - sumL;
            double sumSqR     = sumSqTotal - sumSqL;
            double nR         = static_cast<double>(nIdx - pos);
            double mR         = sumR / nR;
            double varR       = (sumSqR / nR) - (mR * mR);

            // MSE = variance
            double msel = varL;
            double mser = varR;
            // 计算加权增益：parentMetric - (mL*nL + mR*nR)/(nIdx)
            double gain = parentMetric - (msel * nL + mser * nR) / (double)nIdx;

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = thr;
            }
        }

        // 更新线程局部变量：feature 索引记在 bestFeatPerThread
        if (localBestGain > bestGainPerThread[tid]) {
            bestGainPerThread[tid] = localBestGain;
            bestFeatPerThread[tid] = f;
            bestThrPerThread[tid]  = localBestThr;
        }
    };

    // **并行或串行遍历特征**
    if (useParallel) {
        #pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            #pragma omp for schedule(dynamic)
            for (int f = 0; f < D; ++f) {
                processFeature(f, tid);
            }
        }
    } else {
        // 串行遍历所有特征
        for (int f = 0; f < D; ++f) {
            processFeature(f, /*tid=*/0);
        }
    }

    // **归约所有线程的局部最优**
    for (int t = 0; t < maxThreads; ++t) {
        double gain = bestGainPerThread[t];
        if (gain > globalBestGain) {
            globalBestGain  = gain;
            globalBestFeat  = bestFeatPerThread[t];
            globalBestThr   = bestThrPerThread[t];
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}
