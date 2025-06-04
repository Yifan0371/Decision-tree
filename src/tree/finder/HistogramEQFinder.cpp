// =============================================================================
// src/tree/finder/HistogramEQFinder.cpp - 预计算直方图优化版本
// =============================================================================
#include "finder/HistogramEQFinder.hpp"
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <ostream>
#include <numeric>   // 为了使用 std::iota

#include <iostream>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

// 等频直方图专用管理器
static thread_local std::unique_ptr<PrecomputedHistograms> g_eqHistogramManager = nullptr;
static thread_local bool g_eqHistogramInitialized = false;

static PrecomputedHistograms* getEQHistogramManager(int numFeatures) {
    if (!g_eqHistogramInitialized) {
        g_eqHistogramManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_eqHistogramInitialized = true;
    }
    return g_eqHistogramManager.get();
}

std::tuple<int, double, double>
HistogramEQFinder::findBestSplit(const std::vector<double>& X,
                                 int                        D,
                                 const std::vector<double>& y,
                                 const std::vector<int>&    idx,
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const {
    
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    // **核心优化1: 使用等频预计算直方图管理器**
    PrecomputedHistograms* histManager = getEQHistogramManager(D);
    
    // **优化2: 首次调用时预计算等频直方图**
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        std::vector<int> allIndices(y.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // 预计算等频直方图
        histManager->precompute(X, D, y, allIndices, "equal_frequency", bins_);
        isFirstCall = false;
        
        std::cout << "HistogramEQ: Precomputed equal-frequency histograms for " << D 
                  << " features with " << bins_ << " bins" << std::endl;
    }
    
    // **优化3: 快速等频分裂查找**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        X, D, y, idx, parentMetric);
    
    // 如果快速查找失败，使用优化的传统等频方法
    if (bestFeat < 0) {
        return findBestSplitEqualFrequencyOptimized(X, D, y, idx, parentMetric, crit);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **优化的等频分裂查找**: 避免每次重新排序
std::tuple<int, double, double>
HistogramEQFinder::findBestSplitEqualFrequencyOptimized(const std::vector<double>& X,
                                                        int D,
                                                        const std::vector<double>& y,
                                                        const std::vector<int>& idx,
                                                        double parentMetric,
                                                        const ISplitCriterion& crit) const {

    const size_t N = idx.size();
    const int per = std::max(1, static_cast<int>(N) / bins_);
    const double EPS = 1e-12;

    int bestFeat = -1;
    double bestThr = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    // **优化4: 智能并行决策**
    const bool useParallel = (N > 500 && D > 4);

    if (useParallel) {
        #pragma omp parallel
        {
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();

            // **优化5: 线程局部排序缓冲（减少内存分配）**
            std::vector<int> localSorted;
            localSorted.reserve(N);
            std::vector<int> localLeft, localRight;
            localLeft.reserve(N / 2);
            localRight.reserve(N / 2);

            #pragma omp for schedule(dynamic) nowait
            for (int f = 0; f < D; ++f) {
                // **优化6: 单次排序，多次复用**
                localSorted.clear();
                localSorted.assign(idx.begin(), idx.end());
                
                std::sort(localSorted.begin(), localSorted.end(),
                          [&](int a, int b) {
                              return X[a * D + f] < X[b * D + f];
                          });

                if (localSorted.size() < 2) continue;

                // **优化7: 快速等频分割点生成**
                for (size_t pivot = per; pivot < N; pivot += per) {
                    if (pivot >= N - 1) break;
                    
                    double vL = X[localSorted[pivot - 1] * D + f];
                    double vR = X[localSorted[pivot] * D + f];
                    if (std::abs(vR - vL) < EPS) continue;

                    // **优化8: 原地分割，避免vector拷贝**
                    localLeft.clear();
                    localRight.clear();
                    
                    double threshold = 0.5 * (vL + vR);
                    
                    // 快速分割
                    for (size_t i = 0; i < localSorted.size(); ++i) {
                        int sampleIdx = localSorted[i];
                        if (i < pivot) {
                            localLeft.push_back(sampleIdx);
                        } else {
                            localRight.push_back(sampleIdx);
                        }
                    }

                    if (localLeft.empty() || localRight.empty()) continue;

                    // **优化9: 快速统计计算（避免criterion调用）**
                    double leftSum = 0.0, leftSumSq = 0.0;
                    double rightSum = 0.0, rightSumSq = 0.0;
                    
                    for (int idx : localLeft) {
                        double val = y[idx];
                        leftSum += val;
                        leftSumSq += val * val;
                    }
                    
                    for (int idx : localRight) {
                        double val = y[idx];
                        rightSum += val;
                        rightSumSq += val * val;
                    }
                    
                    // 内联MSE计算
                    double leftMSE = leftSumSq / localLeft.size() - 
                                     std::pow(leftSum / localLeft.size(), 2);
                    double rightMSE = rightSumSq / localRight.size() - 
                                      std::pow(rightSum / localRight.size(), 2);
                    
                    double gain = parentMetric - 
                                 (leftMSE * localLeft.size() + rightMSE * localRight.size()) / N;

                    if (gain > localBestGain) {
                        localBestGain = gain;
                        localBestFeat = f;
                        localBestThr = threshold;
                    }
                }
            }

            #pragma omp critical
            {
                if (localBestGain > bestGain) {
                    bestGain = localBestGain;
                    bestFeat = localBestFeat;
                    bestThr = localBestThr;
                }
            }
        }
    } else {
        // **串行优化版本**
        std::vector<int> sortedIdx;
        sortedIdx.reserve(N);
        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(N / 2);
        rightBuf.reserve(N / 2);

        for (int f = 0; f < D; ++f) {
            sortedIdx.assign(idx.begin(), idx.end());
            std::sort(sortedIdx.begin(), sortedIdx.end(),
                      [&](int a, int b) {
                          return X[a * D + f] < X[b * D + f];
                      });

            if (sortedIdx.size() < 2) continue;

            // **优化10: 批量等频分割点评估**
            std::vector<size_t> pivotPoints;
            for (size_t pivot = per; pivot < N; pivot += per) {
                if (pivot < N - 1) {
                    double vL = X[sortedIdx[pivot - 1] * D + f];
                    double vR = X[sortedIdx[pivot] * D + f];
                    if (std::abs(vR - vL) >= EPS) {
                        pivotPoints.push_back(pivot);
                    }
                }
            }
            
            // 批量评估所有有效分割点
            for (size_t pivot : pivotPoints) {
                leftBuf.clear();
                rightBuf.clear();
                
                leftBuf.assign(sortedIdx.begin(), sortedIdx.begin() + pivot);
                rightBuf.assign(sortedIdx.begin() + pivot, sortedIdx.end());

                if (leftBuf.empty() || rightBuf.empty()) continue;

                // 快速MSE计算
                double leftSum = 0.0, leftSumSq = 0.0;
                double rightSum = 0.0, rightSumSq = 0.0;
                
                for (int idx : leftBuf) {
                    double val = y[idx];
                    leftSum += val;
                    leftSumSq += val * val;
                }
                
                for (int idx : rightBuf) {
                    double val = y[idx];
                    rightSum += val;
                    rightSumSq += val * val;
                }
                
                double leftMSE = leftSumSq / leftBuf.size() - 
                                 std::pow(leftSum / leftBuf.size(), 2);
                double rightMSE = rightSumSq / rightBuf.size() - 
                                  std::pow(rightSum / rightBuf.size(), 2);
                
                double gain = parentMetric - 
                             (leftMSE * leftBuf.size() + rightMSE * rightBuf.size()) / N;

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeat = f;
                    double vL = X[sortedIdx[pivot - 1] * D + f];
                    double vR = X[sortedIdx[pivot] * D + f];
                    bestThr = 0.5 * (vL + vR);
                }
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}