// =============================================================================
// src/tree/finder/HistogramEWFinder.cpp - 预计算直方图优化版本
// =============================================================================
#include "finder/HistogramEWFinder.hpp"

// 添加传统优化方法的声明到头文件中
// 在HistogramEWFinder类中添加：
// std::tuple<int, double, double> findBestSplitTraditionalOptimized(
//     const std::vector<double>& X, int D, const std::vector<double>& y,
//     const std::vector<int>& idx, double parentMetric, const ISplitCriterion& crit) const;
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <iostream>
#include <numeric>   // 为了使用 std::iota

#include <ostream>
#ifdef _OPENMP
#include <omp.h>
#endif

// 静态全局直方图管理器（线程安全的单例模式）
static thread_local std::unique_ptr<PrecomputedHistograms> g_histogramManager = nullptr;
static thread_local bool g_histogramInitialized = false;

// 直方图管理器访问函数
static PrecomputedHistograms* getHistogramManager(int numFeatures) {
    if (!g_histogramInitialized) {
        g_histogramManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_histogramInitialized = true;
    }
    return g_histogramManager.get();
}

std::tuple<int, double, double>
HistogramEWFinder::findBestSplit(const std::vector<double>& X,
                                 int                        D,
                                 const std::vector<double>& y,
                                 const std::vector<int>&    idx,
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const {
    
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    // **核心优化1: 使用预计算直方图管理器**
    PrecomputedHistograms* histManager = getHistogramManager(D);
    
    // **优化2: 首次调用时预计算所有直方图**
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        // 创建全数据集索引用于预计算
        std::vector<int> allIndices(y.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // 一次性预计算所有特征的等宽直方图
        histManager->precompute(X, D, y, allIndices, "equal_width", bins_);
        isFirstCall = false;
        
        std::cout << "HistogramEW: Precomputed histograms for " << D 
                  << " features with " << bins_ << " bins" << std::endl;
    }
    
    // **优化3: 使用快速分裂查找，避免重新计算直方图**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        X, D, y, idx, parentMetric);
    
    // 如果快速查找失败，回退到传统方法（但仍然优化）
    if (bestFeat < 0) {
        return findBestSplitTraditionalOptimized(X, D, y, idx, parentMetric, crit);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **优化的传统方法**: 保留作为备选，但仍进行了优化
std::tuple<int, double, double>
HistogramEWFinder::findBestSplitTraditionalOptimized(const std::vector<double>& X,
                                                     int D,
                                                     const std::vector<double>& y,
                                                     const std::vector<int>& idx,
                                                     double parentMetric,
                                                     const ISplitCriterion& crit) const {

    const size_t N = idx.size();
    int globalBestFeat = -1;
    double globalBestThr = 0.0;
    double globalBestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // **优化4: 智能并行策略 - 减少线程创建开销**
    const bool useParallel = (N > 1000 && D > 4);
    
    if (useParallel) {
        #pragma omp parallel
        {
            // 线程局部变量
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();
            
            // **优化5: 线程局部直方图缓冲（避免false sharing）**
            std::vector<int> histCnt(bins_);
            std::vector<double> histSum(bins_);
            std::vector<double> histSumSq(bins_);
            std::vector<int> prefixCnt(bins_);
            std::vector<double> prefixSum(bins_);
            std::vector<double> prefixSumSq(bins_);

            #pragma omp for schedule(dynamic) nowait
            for (int f = 0; f < D; ++f) {
                // **优化6: 快速特征范围计算（避免多次遍历）**
                double vMin = std::numeric_limits<double>::infinity();
                double vMax = -vMin;
                
                for (int i : idx) {
                    double v = X[i * D + f];
                    vMin = std::min(vMin, v);
                    vMax = std::max(vMax, v);
                }
                
                if (std::abs(vMax - vMin) < EPS) continue;

                const double binW = (vMax - vMin) / bins_;

                // **优化7: 快速直方图构建（单次遍历）**
                std::fill(histCnt.begin(), histCnt.end(), 0);
                std::fill(histSum.begin(), histSum.end(), 0.0);
                std::fill(histSumSq.begin(), histSumSq.end(), 0.0);

                for (int i : idx) {
                    const double v = X[i * D + f];
                    int b = static_cast<int>((v - vMin) / binW);
                    if (b == bins_) b--;
                    const double lbl = y[i];

                    histCnt[b] += 1;
                    histSum[b] += lbl;
                    histSumSq[b] += lbl * lbl;
                }

                // **优化8: 向量化前缀计算**
                prefixCnt[0] = histCnt[0];
                prefixSum[0] = histSum[0];
                prefixSumSq[0] = histSumSq[0];
                
                for (int b = 1; b < bins_; ++b) {
                    prefixCnt[b] = prefixCnt[b-1] + histCnt[b];
                    prefixSum[b] = prefixSum[b-1] + histSum[b];
                    prefixSumSq[b] = prefixSumSq[b-1] + histSumSq[b];
                }

                // **优化9: 快速分裂评估（避免重复计算）**
                const double totalSum = prefixSum[bins_-1];
                const double totalSumSq = prefixSumSq[bins_-1];
                
                for (int b = 0; b < bins_ - 1; ++b) {
                    const int leftCnt = prefixCnt[b];
                    const int rightCnt = static_cast<int>(N) - leftCnt;
                    if (leftCnt == 0 || rightCnt == 0) continue;

                    const double leftSum = prefixSum[b];
                    const double leftSumSq = prefixSumSq[b];
                    const double rightSum = totalSum - leftSum;
                    const double rightSumSq = totalSumSq - leftSumSq;

                    // **优化10: 内联MSE计算（减少函数调用）**
                    const double leftMean = leftSum / leftCnt;
                    const double rightMean = rightSum / rightCnt;
                    const double leftMSE = leftSumSq / leftCnt - leftMean * leftMean;
                    const double rightMSE = rightSumSq / rightCnt - rightMean * rightMean;
                    
                    const double gain = parentMetric - (leftMSE * leftCnt + rightMSE * rightCnt) / N;

                    if (gain > localBestGain) {
                        localBestGain = gain;
                        localBestFeat = f;
                        localBestThr = vMin + (b + 0.5) * binW;
                    }
                }
            }
            
            // **优化11: 减少critical区域时间**
            #pragma omp critical
            {
                if (localBestGain > globalBestGain) {
                    globalBestGain = localBestGain;
                    globalBestFeat = localBestFeat;
                    globalBestThr = localBestThr;
                }
            }
        }
    } else {
        // **串行版本 - 小数据集优化**
        std::vector<int> histCnt(bins_);
        std::vector<double> histSum(bins_);
        std::vector<double> histSumSq(bins_);
        
        for (int f = 0; f < D; ++f) {
            // 计算特征范围
            auto [minIt, maxIt] = std::minmax_element(idx.begin(), idx.end(),
                [&](int a, int b) { return X[a * D + f] < X[b * D + f]; });
            
            double vMin = X[*minIt * D + f];
            double vMax = X[*maxIt * D + f];
            
            if (std::abs(vMax - vMin) < EPS) continue;

            const double binW = (vMax - vMin) / bins_;

            // 构建直方图
            std::fill(histCnt.begin(), histCnt.end(), 0);
            std::fill(histSum.begin(), histSum.end(), 0.0);
            std::fill(histSumSq.begin(), histSumSq.end(), 0.0);

            for (int i : idx) {
                const double v = X[i * D + f];
                int b = static_cast<int>((v - vMin) / binW);
                if (b == bins_) b--;
                const double lbl = y[i];

                histCnt[b] += 1;
                histSum[b] += lbl;
                histSumSq[b] += lbl * lbl;
            }

            // 评估分裂点
            double leftSum = 0.0, leftSumSq = 0.0;
            int leftCnt = 0;
            
            for (int b = 0; b < bins_ - 1; ++b) {
                leftSum += histSum[b];
                leftSumSq += histSumSq[b];
                leftCnt += histCnt[b];
                
                const int rightCnt = static_cast<int>(N) - leftCnt;
                if (leftCnt == 0 || rightCnt == 0) continue;
                
                double rightSum = 0.0, rightSumSq = 0.0;
                for (int rb = b + 1; rb < bins_; ++rb) {
                    rightSum += histSum[rb];
                    rightSumSq += histSumSq[rb];
                }

                const double leftMSE = leftSumSq / leftCnt - std::pow(leftSum / leftCnt, 2);
                const double rightMSE = rightSumSq / rightCnt - std::pow(rightSum / rightCnt, 2);
                const double gain = parentMetric - (leftMSE * leftCnt + rightMSE * rightCnt) / N;

                if (gain > globalBestGain) {
                    globalBestGain = gain;
                    globalBestFeat = f;
                    globalBestThr = vMin + (b + 0.5) * binW;
                }
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}