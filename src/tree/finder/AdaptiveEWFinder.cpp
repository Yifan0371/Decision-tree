// =============================================================================
// src/tree/finder/AdaptiveEWFinder.cpp - 预计算直方图优化版本
// =============================================================================
#include "finder/AdaptiveEWFinder.hpp"
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <iostream>

#include <ostream>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

// 自适应等宽直方图管理器
static thread_local std::unique_ptr<PrecomputedHistograms> g_adaptiveEWManager = nullptr;
static thread_local bool g_adaptiveEWInitialized = false;

static PrecomputedHistograms* getAdaptiveEWManager(int numFeatures) {
    if (!g_adaptiveEWInitialized) {
        g_adaptiveEWManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_adaptiveEWInitialized = true;
    }
    return g_adaptiveEWManager.get();
}

// **优化的工具函数**
static double calculateIQRFast(std::vector<double>& values) {
    if (values.size() < 4) return 0.0;
    
    // **优化: 使用nth_element代替完全排序**
    const size_t n = values.size();
    const size_t q1_pos = n / 4;
    const size_t q3_pos = 3 * n / 4;
    
    std::nth_element(values.begin(), values.begin() + q1_pos, values.end());
    double q1 = values[q1_pos];
    
    std::nth_element(values.begin() + q1_pos + 1, values.begin() + q3_pos, values.end());
    double q3 = values[q3_pos];
    
    return q3 - q1;
}

std::tuple<int, double, double>
AdaptiveEWFinder::findBestSplit(const std::vector<double>& data,
                                int                       rowLen,
                                const std::vector<double>&labels,
                                const std::vector<int>&   idx,
                                double                    parentMetric,
                                const ISplitCriterion&    criterion) const {
    
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    // **核心优化1: 使用自适应等宽预计算直方图**
    PrecomputedHistograms* histManager = getAdaptiveEWManager(rowLen);
    
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        std::vector<int> allIndices(labels.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // **优化2: 预计算自适应等宽直方图**
        histManager->precompute(data, rowLen, labels, allIndices, "adaptive_ew", 0);
        isFirstCall = false;
        
        std::cout << "AdaptiveEW: Precomputed adaptive equal-width histograms for " 
                  << rowLen << " features" << std::endl;
    }
    
    // **优化3: 快速自适应分裂查找**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        data, rowLen, labels, idx, parentMetric);
    
    // 备选优化方法
    if (bestFeat < 0) {
        return findBestSplitAdaptiveEWOptimized(data, rowLen, labels, idx, parentMetric, criterion);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **优化的自适应等宽方法**
std::tuple<int, double, double>
AdaptiveEWFinder::findBestSplitAdaptiveEWOptimized(const std::vector<double>& data,
                                                   int rowLen,
                                                   const std::vector<double>& labels,
                                                   const std::vector<int>& idx,
                                                   double parentMetric,
                                                   const ISplitCriterion& criterion) const {

    const size_t N = idx.size();
    int globalBestFeat = -1;
    double globalBestThr = 0.0;
    double globalBestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // **优化4: 智能并行策略**
    const bool useParallel = (N > 1000 && rowLen > 4);

    if (useParallel) {
        #pragma omp parallel
        {
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();

            // **优化5: 线程局部缓冲（减少内存分配和false sharing）**
            std::vector<double> values;
            std::vector<std::vector<int>> buckets;
            std::vector<int> leftBuf, rightBuf;
            values.reserve(N);
            buckets.reserve(128);  // 最大桶数
            leftBuf.reserve(N);
            rightBuf.reserve(N);

            #pragma omp for schedule(dynamic) nowait
            for (int f = 0; f < rowLen; ++f) {
                // **优化6: 单次遍历收集特征值**
                values.clear();
                for (int i : idx) {
                    values.emplace_back(data[i * rowLen + f]);
                }

                if (values.empty()) continue;

                // **优化7: 快速最优桶数计算**
                int optimalBins = calculateOptimalBinsFast(values);
                if (optimalBins < 2) continue;

                auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
                double vMin = *vMinIt;
                double vMax = *vMaxIt;
                if (std::abs(vMax - vMin) < EPS) continue;

                double binW = (vMax - vMin) / optimalBins;

                // **优化8: 快速分桶（避免重复计算）**
                buckets.clear();
                buckets.resize(optimalBins);
                
                for (int i : idx) {
                    double val = data[i * rowLen + f];
                    int b = static_cast<int>((val - vMin) / binW);
                    if (b == optimalBins) b--;
                    buckets[b].push_back(i);
                }

                // **优化9: 批量分裂评估（减少循环开销）**
                leftBuf.clear();
                
                for (int b = 0; b < optimalBins - 1; ++b) {
                    // 累积左侧桶
                    leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
                    if (leftBuf.empty()) continue;

                    size_t leftN = leftBuf.size();
                    size_t rightN = N - leftN;
                    if (rightN == 0) break;

                    // **优化10: 快速MSE计算（内联，避免函数调用）**
                    double leftSum = 0.0, leftSumSq = 0.0;
                    double rightSum = 0.0, rightSumSq = 0.0;
                    
                    for (int idx : leftBuf) {
                        double val = labels[idx];
                        leftSum += val;
                        leftSumSq += val * val;
                    }
                    
                    // 快速计算右侧统计（避免构建rightBuf）
                    for (int k = b + 1; k < optimalBins; ++k) {
                        for (int idx : buckets[k]) {
                            double val = labels[idx];
                            rightSum += val;
                            rightSumSq += val * val;
                        }
                    }
                    
                    if (rightN > 0) {
                        double leftMSE = leftSumSq / leftN - std::pow(leftSum / leftN, 2);
                        double rightMSE = rightSumSq / rightN - std::pow(rightSum / rightN, 2);
                        double gain = parentMetric - (leftMSE * leftN + rightMSE * rightN) / N;

                        if (gain > localBestGain) {
                            localBestGain = gain;
                            localBestFeat = f;
                            localBestThr = vMin + binW * (b + 1);
                        }
                    }
                }
            }

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
        // **串行优化版本**
        std::vector<double> values;
        std::vector<std::vector<int>> buckets;
        values.reserve(N);
        buckets.reserve(128);

        for (int f = 0; f < rowLen; ++f) {
            values.clear();
            for (int i : idx) {
                values.emplace_back(data[i * rowLen + f]);
            }

            if (values.empty()) continue;

            int optimalBins = calculateOptimalBinsFast(values);
            if (optimalBins < 2) continue;

            auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
            double vMin = *vMinIt;
            double vMax = *vMaxIt;
            if (std::abs(vMax - vMin) < EPS) continue;

            double binW = (vMax - vMin) / optimalBins;

            // 分桶
            buckets.clear();
            buckets.resize(optimalBins);
            for (int i : idx) {
                double val = data[i * rowLen + f];
                int b = static_cast<int>((val - vMin) / binW);
                if (b == optimalBins) b--;
                buckets[b].push_back(i);
            }

            // 评估分裂
            std::vector<int> leftBuf;
            leftBuf.reserve(N);

            for (int b = 0; b < optimalBins - 1; ++b) {
                leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
                if (leftBuf.empty()) continue;

                size_t leftN = leftBuf.size();
                size_t rightN = N - leftN;
                if (rightN == 0) break;

                // 快速统计计算
                double leftSum = 0.0, leftSumSq = 0.0;
                double rightSum = 0.0, rightSumSq = 0.0;
                
                for (int idx : leftBuf) {
                    double val = labels[idx];
                    leftSum += val;
                    leftSumSq += val * val;
                }
                
                for (int k = b + 1; k < optimalBins; ++k) {
                    for (int idx : buckets[k]) {
                        double val = labels[idx];
                        rightSum += val;
                        rightSumSq += val * val;
                    }
                }
                
                if (rightN > 0) {
                    double leftMSE = leftSumSq / leftN - std::pow(leftSum / leftN, 2);
                    double rightMSE = rightSumSq / rightN - std::pow(rightSum / rightN, 2);
                    double gain = parentMetric - (leftMSE * leftN + rightMSE * rightN) / N;

                    if (gain > globalBestGain) {
                        globalBestGain = gain;
                        globalBestFeat = f;
                        globalBestThr = vMin + binW * (b + 1);
                    }
                }
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}

// **优化的最优桶数计算**
int AdaptiveEWFinder::calculateOptimalBinsFast(const std::vector<double>& values) const {
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
        // **优化: 使用快速IQR计算**
        std::vector<double> valuesCopy = values;  // 需要可修改的副本
        double iqr = calculateIQRFast(valuesCopy);
        if (iqr > 0.0) {
            auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
            double h = 2.0 * iqr / std::cbrt(n);
            bins = static_cast<int>(std::ceil((*maxIt - *minIt) / h));
        }
    }

    return std::clamp(bins, minBins_, maxBins_);
}

// **保留旧接口的兼容性**
double AdaptiveEWFinder::calculateIQR(std::vector<double> values) const {
    return calculateIQRFast(values);
}