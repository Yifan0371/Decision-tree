// =============================================================================
// include/histogram/PrecomputedHistograms.hpp - 预计算直方图优化
// =============================================================================
#pragma once

#include <vector>
#include <string>
#include <algorithm> 
#include <memory>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * 直方图桶结构 - 优化的内存布局
 */
struct HistogramBin {
    std::vector<int> sampleIndices;     // 该桶中的样本索引
    double sum = 0.0;                   // 标签值和
    double sumSq = 0.0;                 // 标签值平方和
    int count = 0;                      // 样本数量
    double binStart = 0.0;              // 桶的起始值
    double binEnd = 0.0;                // 桶的结束值
    
    // 快速统计更新
    void addSample(int idx, double label) {
        sampleIndices.push_back(idx);
        sum += label;
        sumSq += label * label;
        ++count;
    }
    
    void removeSample(int idx, double label) {
        auto it = std::find(sampleIndices.begin(), sampleIndices.end(), idx);
        if (it != sampleIndices.end()) {
            sampleIndices.erase(it);
            sum -= label;
            sumSq -= label * label;
            --count;
        }
    }
    
    double getMSE() const {
        if (count == 0) return 0.0;
        double mean = sum / count;
        return sumSq / count - mean * mean;
    }
};

/**
 * 特征直方图 - 单个特征的所有桶
 */
struct FeatureHistogram {
    int featureIndex;
    std::vector<HistogramBin> bins;
    std::vector<double> binBoundaries;  // 桶边界值
    std::string binningType;            // "equal_width", "equal_frequency", "adaptive_ew", "adaptive_eq"
    
    // 前缀统计数组（用于快速范围查询）
    std::vector<double> prefixSum;
    std::vector<double> prefixSumSq; 
    std::vector<int> prefixCount;
    
    void updatePrefixArrays() {
        int numBins = static_cast<int>(bins.size());
        prefixSum.resize(numBins + 1, 0.0);
        prefixSumSq.resize(numBins + 1, 0.0);
        prefixCount.resize(numBins + 1, 0);
        
        for (int i = 0; i < numBins; ++i) {
            prefixSum[i + 1] = prefixSum[i] + bins[i].sum;
            prefixSumSq[i + 1] = prefixSumSq[i] + bins[i].sumSq;
            prefixCount[i + 1] = prefixCount[i] + bins[i].count;
        }
    }
    
    // 快速计算范围[startBin, endBin)的统计量
    void getRangeStats(int startBin, int endBin, double& sum, double& sumSq, int& count) const {
        sum = prefixSum[endBin] - prefixSum[startBin];
        sumSq = prefixSumSq[endBin] - prefixSumSq[startBin];
        count = prefixCount[endBin] - prefixCount[startBin];
    }
};

/**
 * 预计算直方图管理器 - 核心优化类
 */
class PrecomputedHistograms {
public:
    explicit PrecomputedHistograms(int numFeatures) : numFeatures_(numFeatures) {
        histograms_.resize(numFeatures);
    }
    
    /**
     * 预处理阶段：一次性计算所有特征的直方图
     */
    void precompute(const std::vector<double>& data,
                    int rowLength,
                    const std::vector<double>& labels,
                    const std::vector<int>& sampleIndices,
                    const std::string& defaultBinningType = "equal_width",
                    int defaultBins = 64);
    
    /**
     * 为特定特征设置自定义分箱参数
     */
    void setFeatureBinning(int featureIndex, 
                          const std::string& binningType, 
                          int numBins,
                          const std::vector<double>& customBoundaries = {});
    
    /**
     * 快速分裂查找 - 基于预计算直方图
     */
    std::tuple<int, double, double> findBestSplitFast(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,
        const std::vector<int>& nodeIndices,
        double parentMetric,
        const std::vector<int>& candidateFeatures = {}) const;
    
    /**
     * 子节点直方图快速更新 - 核心优化
     */
    void updateChildHistograms(int featureIndex,
                              double splitThreshold,
                              const std::vector<int>& parentIndices,
                              std::vector<int>& leftIndices,
                              std::vector<int>& rightIndices,
                              FeatureHistogram& leftHist,
                              FeatureHistogram& rightHist) const;
    
    /**
     * 获取特征的直方图
     */
    const FeatureHistogram& getFeatureHistogram(int featureIndex) const {
        return histograms_[featureIndex];
    }
    
    FeatureHistogram& getFeatureHistogram(int featureIndex) {
        return histograms_[featureIndex];
    }
    
    /**
     * 内存使用统计
     */
    size_t getMemoryUsage() const;
    
    /**
     * 性能统计
     */
    struct PerformanceStats {
        double precomputeTimeMs = 0.0;
        double splitFindTimeMs = 0.0;
        double histogramUpdateTimeMs = 0.0;
        int totalSplitQueries = 0;
        int totalHistogramUpdates = 0;
    };
    
    const PerformanceStats& getPerformanceStats() const { return stats_; }
    void resetPerformanceStats() { stats_ = PerformanceStats{}; }

private:
    int numFeatures_;
    std::vector<FeatureHistogram> histograms_;
    mutable PerformanceStats stats_;
    
    // 内部辅助方法
    void computeEqualWidthBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              int numBins);
    
    void computeEqualFrequencyBins(int featureIndex,
                                  const std::vector<double>& featureValues,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int numBins);
    
    void computeAdaptiveEWBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              const std::string& rule = "sturges");
    
    void computeAdaptiveEQBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              int minSamplesPerBin = 5,
                              double variabilityThreshold = 0.1);
    
    // 桶分配辅助函数
    int findBin(const FeatureHistogram& hist, double value) const;
    
    // 并行优化辅助函数
    void parallelBinConstruction(int featureIndex,
                                const std::vector<double>& featureValues,
                                const std::vector<double>& labels,
                                const std::vector<int>& indices,
                                const std::vector<double>& boundaries);
};

/**
 * 直方图缓存管理器 - 用于节点级别的缓存
 */
class HistogramCache {
public:
    explicit HistogramCache(int maxCacheSize = 1000) : maxCacheSize_(maxCacheSize) {}
    
    bool hasHistogram(const std::vector<int>& nodeIndices, int featureIndex) const;
    
    const FeatureHistogram& getHistogram(const std::vector<int>& nodeIndices, int featureIndex) const;
    
    void cacheHistogram(const std::vector<int>& nodeIndices, 
                       int featureIndex,
                       const FeatureHistogram& histogram);
    
    void clear() { cache_.clear(); }
    
    size_t size() const { return cache_.size(); }

private:
    int maxCacheSize_;
    mutable std::unordered_map<std::string, FeatureHistogram> cache_;
    
    std::string generateKey(const std::vector<int>& nodeIndices, int featureIndex) const;
    void evictOldEntries();
};