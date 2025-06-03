// src/tree/finder/RandomSplitFinder.cpp - OpenMP并行版本
#include "finder/RandomSplitFinder.hpp"
#include <limits>
#include <random>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double>
RandomSplitFinder::findBestSplit(const std::vector<double>& X,
                                 int                          D,
                                 const std::vector<double>&  y,
                                 const std::vector<int>&     idx,
                                 double                      parentMetric,
                                 const ISplitCriterion&      crit) const
{
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    int    globalBestFeat  = -1;
    double globalBestThr   = 0.0;
    double globalBestGain  = -std::numeric_limits<double>::infinity();

    // **并行化特征遍历**
    #pragma omp parallel
    {
        // 线程局部变量
        int    localBestFeat = -1;
        double localBestThr  = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        // 线程局部随机数生成器（避免竞争）
        thread_local std::mt19937 localGen(gen_() + omp_get_thread_num());
        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        
        // 线程局部缓冲区
        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(idx.size());
        rightBuf.reserve(idx.size());

        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < D; ++f) {
            /* --- 计算该特征的最小/最大值 --- */
            double vMin =  std::numeric_limits<double>::infinity();
            double vMax = -vMin;
            for (int i : idx) {
                double v = X[i * D + f];
                vMin = std::min(vMin, v);
                vMax = std::max(vMax, v);
            }
            if (std::fabs(vMax - vMin) < 1e-12) continue;

            /* --- 对该特征尝试 k_ 个随机阈值 --- */
            for (int r = 0; r < k_; ++r) {
                double thr = vMin + uni01(localGen) * (vMax - vMin);

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

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeat = f;
                    localBestThr  = thr;
                }
            }
        }
        
        /* --- 线程间归约：更新全局最佳结果 --- */
        #pragma omp critical
        {
            if (localBestGain > globalBestGain) {
                globalBestGain = localBestGain;
                globalBestFeat = localBestFeat;
                globalBestThr  = localBestThr;
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}