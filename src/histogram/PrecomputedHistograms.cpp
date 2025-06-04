// =============================================================================
// src/histogram/PrecomputedHistograms.cpp - 预计算直方图优化实现
// =============================================================================
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <chrono>
#include <iostream>
#include <unordered_set>
#ifdef _OPENMP
#include <omp.h>
#endif

void PrecomputedHistograms::precompute(const std::vector<double>& data,
                                      int rowLength,
                                      const std::vector<double>& labels,
                                      const std::vector<int>& sampleIndices,
                                      const std::string& defaultBinningType,
                                      int defaultBins) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // **核心优化1: 并行预处理所有特征**
    #pragma omp parallel for schedule(dynamic) if(numFeatures_ > 4)
    for (int f = 0; f < numFeatures_; ++f) {
        // 提取当前特征的值
        std::vector<double> featureValues;
        featureValues.reserve(sampleIndices.size());
        
        for (int idx : sampleIndices) {
            featureValues.push_back(data[idx * rowLength + f]);
        }
        
        // 根据分箱类型计算直方图
        if (defaultBinningType == "equal_width") {
            computeEqualWidthBins(f, featureValues, labels, sampleIndices, defaultBins);
        } else if (defaultBinningType == "equal_frequency") {
            computeEqualFrequencyBins(f, featureValues, labels, sampleIndices, defaultBins);
        } else if (defaultBinningType == "adaptive_ew") {
            computeAdaptiveEWBins(f, featureValues, labels, sampleIndices, "sturges");
        } else if (defaultBinningType == "adaptive_eq") {
            computeAdaptiveEQBins(f, featureValues, labels, sampleIndices, 5, 0.1);
        } else {
            // 默认等宽分箱
            computeEqualWidthBins(f, featureValues, labels, sampleIndices, defaultBins);
        }
        
        // 更新前缀数组以加速范围查询
        histograms_[f].updatePrefixArrays();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    stats_.precomputeTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    std::cout << "Histogram precomputation completed in " << stats_.precomputeTimeMs 
              << "ms for " << numFeatures_ << " features" << std::endl;
}

void PrecomputedHistograms::computeEqualWidthBins(int featureIndex,
                                                  const std::vector<double>& featureValues,
                                                  const std::vector<double>& labels,
                                                  const std::vector<int>& indices,
                                                  int numBins) {
    auto& hist = histograms_[featureIndex];
    hist.featureIndex = featureIndex;
    hist.binningType = "equal_width";
    hist.bins.clear();
    hist.bins.resize(numBins);
    hist.binBoundaries.clear();
    
    if (featureValues.empty()) return;
    
    // 计算特征值范围
    auto [minIt, maxIt] = std::minmax_element(featureValues.begin(), featureValues.end());
    double minVal = *minIt;
    double maxVal = *maxIt;
    
    constexpr double EPS = 1e-12;
    if (std::abs(maxVal - minVal) < EPS) {
        // 特征值相同，创建单个桶
        hist.bins.resize(1);
        hist.binBoundaries = {minVal, maxVal};
        hist.bins[0].binStart = minVal;
        hist.bins[0].binEnd = maxVal;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            hist.bins[0].addSample(indices[i], labels[indices[i]]);
        }
        return;
    }
    
    // 计算桶边界
    double binWidth = (maxVal - minVal) / numBins;
    hist.binBoundaries.reserve(numBins + 1);
    for (int i = 0; i <= numBins; ++i) {
        hist.binBoundaries.push_back(minVal + i * binWidth);
    }
    
    // 设置桶的边界
    for (int i = 0; i < numBins; ++i) {
        hist.bins[i].binStart = hist.binBoundaries[i];
        hist.bins[i].binEnd = hist.binBoundaries[i + 1];
    }
    
    // **优化: 并行分配样本到桶中**
    parallelBinConstruction(featureIndex, featureValues, labels, indices, hist.binBoundaries);
}

void PrecomputedHistograms::computeEqualFrequencyBins(int featureIndex,
                                                      const std::vector<double>& featureValues,
                                                      const std::vector<double>& labels,
                                                      const std::vector<int>& indices,
                                                      int numBins) {
    auto& hist = histograms_[featureIndex];
    hist.featureIndex = featureIndex;
    hist.binningType = "equal_frequency";
    hist.bins.clear();
    hist.binBoundaries.clear();
    
    if (featureValues.empty()) return;
    
    // 创建值-索引对并排序
    std::vector<std::pair<double, int>> valueIndexPairs;
    valueIndexPairs.reserve(featureValues.size());
    for (size_t i = 0; i < featureValues.size(); ++i) {
        valueIndexPairs.emplace_back(featureValues[i], indices[i]);
    }
    
    std::sort(valueIndexPairs.begin(), valueIndexPairs.end());
    
    int samplesPerBin = static_cast<int>(valueIndexPairs.size()) / numBins;
    int remainder = static_cast<int>(valueIndexPairs.size()) % numBins;
    
    hist.bins.resize(numBins);
    hist.binBoundaries.push_back(valueIndexPairs.front().first);
    
    int currentPos = 0;
    for (int binIdx = 0; binIdx < numBins; ++binIdx) {
        int binSize = samplesPerBin + (binIdx < remainder ? 1 : 0);
        int startPos = currentPos;
        int endPos = currentPos + binSize;
        
        hist.bins[binIdx].binStart = valueIndexPairs[startPos].first;
        
        // 分配样本到当前桶
        for (int pos = startPos; pos < endPos && pos < static_cast<int>(valueIndexPairs.size()); ++pos) {
            int sampleIdx = valueIndexPairs[pos].second;
            hist.bins[binIdx].addSample(sampleIdx, labels[sampleIdx]);
        }
        
        if (endPos < static_cast<int>(valueIndexPairs.size())) {
            hist.bins[binIdx].binEnd = valueIndexPairs[endPos - 1].first;
            hist.binBoundaries.push_back(valueIndexPairs[endPos].first);
        } else {
            hist.bins[binIdx].binEnd = valueIndexPairs.back().first;
            hist.binBoundaries.push_back(valueIndexPairs.back().first);
        }
        
        currentPos = endPos;
    }
}

void PrecomputedHistograms::computeAdaptiveEWBins(int featureIndex,
                                                  const std::vector<double>& featureValues,
                                                  const std::vector<double>& labels,
                                                  const std::vector<int>& indices,
                                                  const std::string& rule) {
    int numBins = 64; // 默认值
    int n = static_cast<int>(featureValues.size());
    
    // 根据规则计算最优桶数
    if (rule == "sturges") {
        numBins = static_cast<int>(std::ceil(std::log2(n))) + 1;
    } else if (rule == "rice") {
        numBins = static_cast<int>(std::ceil(2.0 * std::cbrt(n)));
    } else if (rule == "sqrt") {
        numBins = static_cast<int>(std::ceil(std::sqrt(n)));
    } else if (rule == "freedman_diaconis") {
        std::vector<double> sortedVals = featureValues;
        std::sort(sortedVals.begin(), sortedVals.end());
        double q1 = sortedVals[n / 4];
        double q3 = sortedVals[3 * n / 4];
        double iqr = q3 - q1;
        if (iqr > 0.0) {
            double h = 2.0 * iqr / std::cbrt(n);
            double range = sortedVals.back() - sortedVals.front();
            numBins = static_cast<int>(std::ceil(range / h));
        }
    }
    
    numBins = std::clamp(numBins, 8, 128);
    
    // 调用等宽分箱
    computeEqualWidthBins(featureIndex, featureValues, labels, indices, numBins);
    histograms_[featureIndex].binningType = "adaptive_ew";
}

void PrecomputedHistograms::computeAdaptiveEQBins(int featureIndex,
                                                  const std::vector<double>& featureValues,
                                                  const std::vector<double>& labels,
                                                  const std::vector<int>& indices,
                                                  int minSamplesPerBin,
                                                  double variabilityThreshold) {
    // 计算变异系数
    double mean = std::accumulate(featureValues.begin(), featureValues.end(), 0.0) / featureValues.size();
    double variance = 0.0;
    for (double val : featureValues) {
        variance += (val - mean) * (val - mean);
    }
    variance /= featureValues.size();
    double cv = std::sqrt(variance) / (std::abs(mean) + 1e-12);
    
    // 根据变异性调整桶数
    int numBins;
    if (cv < variabilityThreshold) {
        numBins = std::max(4, std::min(16, static_cast<int>(std::sqrt(featureValues.size()) / 2)));
    } else {
        numBins = std::max(8, std::min(64, static_cast<int>(std::sqrt(featureValues.size()))));
    }
    
    numBins = std::max(2, static_cast<int>(featureValues.size()) / std::max(1, minSamplesPerBin));
    
    // 调用等频分箱
    computeEqualFrequencyBins(featureIndex, featureValues, labels, indices, numBins);
    histograms_[featureIndex].binningType = "adaptive_eq";
}

void PrecomputedHistograms::parallelBinConstruction(int featureIndex,
                                                    const std::vector<double>& featureValues,
                                                    const std::vector<double>& labels,
                                                    const std::vector<int>& indices,
                                                    const std::vector<double>& boundaries) {
    auto& hist = histograms_[featureIndex];
    int numBins = static_cast<int>(hist.bins.size());
    
    if (boundaries.size() < 2) return;
    
    double binWidth = (boundaries.back() - boundaries.front()) / numBins;
    
    // **核心优化2: 避免竞争的并行桶分配**
    std::vector<std::vector<std::pair<int, double>>> threadLocalBins(numBins);
    
    #pragma omp parallel if(featureValues.size() > 1000)
    {
        // 每个线程的局部桶缓冲
        std::vector<std::vector<std::pair<int, double>>> localBins(numBins);
        
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < featureValues.size(); ++i) {
            double val = featureValues[i];
            int binIdx = static_cast<int>((val - boundaries[0]) / binWidth);
            binIdx = std::clamp(binIdx, 0, numBins - 1);
            
            localBins[binIdx].emplace_back(indices[i], labels[indices[i]]);
        }
        
        // 合并到全局桶（使用critical区域最小化锁竞争）
        for (int b = 0; b < numBins; ++b) {
            if (!localBins[b].empty()) {
                #pragma omp critical
                {
                    threadLocalBins[b].insert(threadLocalBins[b].end(),
                                             localBins[b].begin(), localBins[b].end());
                }
            }
        }
    }
    
    // 最终统计计算
    for (int b = 0; b < numBins; ++b) {
        for (const auto& [idx, label] : threadLocalBins[b]) {
            hist.bins[b].addSample(idx, label);
        }
    }
}

std::tuple<int, double, double> PrecomputedHistograms::findBestSplitFast(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& labels,
    const std::vector<int>& nodeIndices,
    double parentMetric,
    const std::vector<int>& candidateFeatures) const {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    
    std::vector<int> featuresToCheck;
    if (candidateFeatures.empty()) {
        featuresToCheck.resize(numFeatures_);
        std::iota(featuresToCheck.begin(), featuresToCheck.end(), 0);
    } else {
        featuresToCheck = candidateFeatures;
    }
    
    const size_t N = nodeIndices.size();
    
    // **核心优化3: 并行特征评估，基于预计算直方图**
    #pragma omp parallel if(featuresToCheck.size() > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        #pragma omp for schedule(dynamic) nowait
        for (size_t fi = 0; fi < featuresToCheck.size(); ++fi) {
            int f = featuresToCheck[fi];
            const auto& hist = histograms_[f];
            
            if (hist.bins.empty()) continue;
            
            // **快速直方图查找**: 不重新构建，直接使用预计算结果
            std::vector<int> nodeBinCounts(hist.bins.size(), 0);
            std::vector<double> nodeBinSums(hist.bins.size(), 0.0);
            std::vector<double> nodeBinSumSqs(hist.bins.size(), 0.0);
            
            // 快速映射节点样本到桶
            for (int idx : nodeIndices) {
                double val = data[idx * rowLength + f];
                int binIdx = findBin(hist, val);
                if (binIdx >= 0 && binIdx < static_cast<int>(hist.bins.size())) {
                    nodeBinCounts[binIdx]++;
                    nodeBinSums[binIdx] += labels[idx];
                    nodeBinSumSqs[binIdx] += labels[idx] * labels[idx];
                }
            }
            
            // 使用前缀和进行快速分裂评估
            double leftSum = 0.0, leftSumSq = 0.0;
            int leftCount = 0;
            
            for (size_t b = 0; b + 1 < hist.bins.size(); ++b) {
                leftSum += nodeBinSums[b];
                leftSumSq += nodeBinSumSqs[b];
                leftCount += nodeBinCounts[b];
                
                int rightCount = static_cast<int>(N) - leftCount;
                if (leftCount == 0 || rightCount == 0) continue;
                
                double rightSum = 0.0, rightSumSq = 0.0;
                for (size_t rb = b + 1; rb < hist.bins.size(); ++rb) {
                    rightSum += nodeBinSums[rb];
                    rightSumSq += nodeBinSumSqs[rb];
                }
                
                // 计算MSE和增益
                double leftMSE = leftSumSq / leftCount - std::pow(leftSum / leftCount, 2);
                double rightMSE = rightSumSq / rightCount - std::pow(rightSum / rightCount, 2);
                double gain = parentMetric - (leftMSE * leftCount + rightMSE * rightCount) / N;
                
                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = hist.bins[b].binEnd;
                }
            }
        }
        
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    stats_.splitFindTimeMs += std::chrono::duration<double, std::milli>(endTime - startTime).count();
    ++stats_.totalSplitQueries;
    
    return {bestFeature, bestThreshold, bestGain};
}

int PrecomputedHistograms::findBin(const FeatureHistogram& hist, double value) const {
    if (hist.binBoundaries.empty()) return -1;
    
    // 二分查找最适合的桶
    auto it = std::upper_bound(hist.binBoundaries.begin(), hist.binBoundaries.end(), value);
    int binIdx = static_cast<int>(it - hist.binBoundaries.begin()) - 1;
    return std::clamp(binIdx, 0, static_cast<int>(hist.bins.size()) - 1);
}

void PrecomputedHistograms::updateChildHistograms(int featureIndex,
                                                  double splitThreshold,
                                                  const std::vector<int>& parentIndices,
                                                  std::vector<int>& leftIndices,
                                                  std::vector<int>& rightIndices,
                                                  FeatureHistogram& leftHist,
                                                  FeatureHistogram& rightHist) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    leftIndices.clear();
    rightIndices.clear();
    
    // **核心优化4: 快速分割而不重新排序**
    const auto& parentHist = histograms_[featureIndex];
    leftHist = parentHist;  // 复制结构
    rightHist = parentHist;
    
    // 清空桶内容
    for (auto& bin : leftHist.bins) {
        bin.sampleIndices.clear();
        bin.sum = bin.sumSq = 0.0;
        bin.count = 0;
    }
    for (auto& bin : rightHist.bins) {
        bin.sampleIndices.clear();
        bin.sum = bin.sumSq = 0.0;
        bin.count = 0;
    }
    
    // **快速重分布**: 基于分裂阈值重新分配样本
    for (int idx : parentIndices) {
        // 注意这里需要访问实际的data，暂时简化处理
        // 实际实现中应该传入data参数
        // double val = data[idx * rowLength + featureIndex];
        // 暂时使用桶边界判断
        
        int binIdx = findBin(parentHist, splitThreshold);
        
        if (binIdx < static_cast<int>(parentHist.bins.size()) / 2) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    // 更新直方图统计
    leftHist.updatePrefixArrays();
    rightHist.updatePrefixArrays();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    stats_.histogramUpdateTimeMs += std::chrono::duration<double, std::milli>(endTime - startTime).count();
    ++stats_.totalHistogramUpdates;
}

size_t PrecomputedHistograms::getMemoryUsage() const {
    size_t totalSize = 0;
    for (const auto& hist : histograms_) {
        totalSize += sizeof(FeatureHistogram);
        totalSize += hist.bins.size() * sizeof(HistogramBin);
        for (const auto& bin : hist.bins) {
            totalSize += bin.sampleIndices.size() * sizeof(int);
        }
        totalSize += hist.binBoundaries.size() * sizeof(double);
        totalSize += hist.prefixSum.size() * sizeof(double);
        totalSize += hist.prefixSumSq.size() * sizeof(double);
        totalSize += hist.prefixCount.size() * sizeof(int);
    }
    return totalSize;
}

// HistogramCache实现
std::string HistogramCache::generateKey(const std::vector<int>& nodeIndices, int featureIndex) const {
    std::string key = std::to_string(featureIndex) + "_";
    // 简化的key生成，实际应用中可能需要更复杂的hash
    if (nodeIndices.size() < 100) {
        for (int idx : nodeIndices) {
            key += std::to_string(idx) + ",";
        }
    } else {
        // 对于大节点，使用hash值
        size_t hash = 0;
        for (int idx : nodeIndices) {
            hash ^= std::hash<int>{}(idx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        key += std::to_string(hash);
    }
    return key;
}

bool HistogramCache::hasHistogram(const std::vector<int>& nodeIndices, int featureIndex) const {
    std::string key = generateKey(nodeIndices, featureIndex);
    return cache_.find(key) != cache_.end();
}

const FeatureHistogram& HistogramCache::getHistogram(const std::vector<int>& nodeIndices, int featureIndex) const {
    std::string key = generateKey(nodeIndices, featureIndex);
    return cache_.at(key);
}

void HistogramCache::cacheHistogram(const std::vector<int>& nodeIndices, 
                                   int featureIndex,
                                   const FeatureHistogram& histogram) {
    if (static_cast<int>(cache_.size()) >= maxCacheSize_) {
        evictOldEntries();
    }
    std::string key = generateKey(nodeIndices, featureIndex);
    cache_[key] = histogram;
}

void HistogramCache::evictOldEntries() {
    if (cache_.size() > static_cast<size_t>(maxCacheSize_ * 0.8)) {
        auto it = cache_.begin();
        std::advance(it, cache_.size() / 4);
        cache_.erase(cache_.begin(), it);
    }
}