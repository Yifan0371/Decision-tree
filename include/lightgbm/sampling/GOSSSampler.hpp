// =============================================================================
// include/lightgbm/sampling/GOSSSampler.hpp
// 深度 OpenMP 并行优化版本（阈值调整、减少重复分配、并行排序条件限定）
// =============================================================================
#pragma once

#include <vector>
#include <random>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

/** GOSS(Gradient-based One-Side Sampling)采样器 */
class GOSSSampler {
public:
    explicit GOSSSampler(double topRate = 0.2, double otherRate = 0.1, uint32_t seed = 42)
        : topRate_(topRate), otherRate_(otherRate), gen_(seed) {}

    /** 
     * 执行 GOSS 采样
     * @param gradients 梯度数组
     * @param sampleIndices 输出：采样后的样本索引
     * @param sampleWeights 输出：采样权重(小梯度样本需要放大权重)
     */
    void sample(const std::vector<double>& gradients,
                std::vector<int>& sampleIndices,
                std::vector<double>& sampleWeights) const;

    /** 
     * 带性能监控的 GOSS 采样
     */
    void sampleWithTiming(const std::vector<double>& gradients,
                          std::vector<int>& sampleIndices,
                          std::vector<double>& sampleWeights,
                          double& samplingTimeMs) const;

    /** 
     * 自适应 GOSS 采样
     */
    void adaptiveSample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;

    /** 采样统计信息 */
    struct SamplingStats {
        size_t totalSamples;        // 总样本数
        size_t selectedSamples;     // 选中样本数
        double samplingRatio;       // 采样率
        double effectiveWeightSum;  // 有效权重和
        double maxGradient;         // 最大梯度值
        double minGradient;         // 最小梯度值
    };

    /** 
     * 获取采样统计信息
     */
    SamplingStats getSamplingStats(const std::vector<double>& gradients,
                                   const std::vector<int>& sampleIndices,
                                   const std::vector<double>& sampleWeights) const;

    /** 获取当前参数 */
    double getTopRate() const { return topRate_; }
    double getOtherRate() const { return otherRate_; }

    /** 更新采样参数 */
    void updateRates(double topRate, double otherRate) {
        topRate_ = topRate;
        otherRate_ = otherRate;
    }

    /** 获取推荐的并行阈值 */
    static size_t getParallelThreshold() { return 10000; }
    /** 计算理论采样率 */
    double getTheoreticalSamplingRatio() const {
        return topRate_ + (1.0 - topRate_) * otherRate_;
    }

private:
    double topRate_;      // 大梯度保留比例
    double otherRate_;    // 小梯度采样比例  
    mutable std::mt19937 gen_;

    /** 大数据集的并行 GOSS 采样 */
    void sampleParallel(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;

    /** 小数据集的串行 GOSS 采样 */
    void sampleSerial(const std::vector<double>& gradients,
                      std::vector<int>& sampleIndices,
                      std::vector<double>& sampleWeights) const;

    /** 验证采样参数 */
    bool validateParameters() const {
        return topRate_ > 0.0 && topRate_ < 1.0 &&
               otherRate_ > 0.0 && otherRate_ < 1.0 &&
               (topRate_ + otherRate_) <= 1.0;
    }
};


