// =============================================================================
// include/lightgbm/sampling/GOSSSampler.hpp - 添加并行方法声明
// =============================================================================
#ifndef LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP
#define LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP

#include <vector>
#include <random>
#include <chrono>
#include <limits>

/** GOSS(Gradient-based One-Side Sampling)采样器 - 深度OpenMP并行优化版本 */
class GOSSSampler {
public:
    explicit GOSSSampler(double topRate = 0.2, double otherRate = 0.1, uint32_t seed = 42)
        : topRate_(topRate), otherRate_(otherRate), gen_(seed) {}
    
    /** 
     * 执行GOSS采样
     * @param gradients 梯度数组
     * @param sampleIndices 输出：采样后的样本索引
     * @param sampleWeights 输出：采样权重(小梯度样本需要放大权重)
     */
    void sample(const std::vector<double>& gradients,
                std::vector<int>& sampleIndices,
                std::vector<double>& sampleWeights) const;
    
    // =============================================
    // 新增：高级并行采样方法
    // =============================================
    
    /** 
     * 带性能监控的GOSS采样
     * @param gradients 梯度数组
     * @param sampleIndices 输出：采样后的样本索引
     * @param sampleWeights 输出：采样权重
     * @param samplingTimeMs 输出：采样耗时（毫秒）
     */
    void sampleWithTiming(const std::vector<double>& gradients,
                          std::vector<int>& sampleIndices,
                          std::vector<double>& sampleWeights,
                          double& samplingTimeMs) const;
    
    /** 
     * 自适应GOSS采样
     * 根据梯度分布自动调整topRate和otherRate参数
     * @param gradients 梯度数组
     * @param sampleIndices 输出：采样后的样本索引
     * @param sampleWeights 输出：采样权重
     */
    void adaptiveSample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;
    
    // =============================================
    // 采样统计和分析
    // =============================================
    
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
     * @param gradients 原始梯度
     * @param sampleIndices 采样索引
     * @param sampleWeights 采样权重
     * @return 采样统计信息
     */
    SamplingStats getSamplingStats(const std::vector<double>& gradients,
                                   const std::vector<int>& sampleIndices,
                                   const std::vector<double>& sampleWeights) const;
    
    // =============================================
    // 配置和调优
    // =============================================
    
    /** 获取当前参数 */
    double getTopRate() const { return topRate_; }
    double getOtherRate() const { return otherRate_; }
    
    /** 更新采样参数 */
    void updateRates(double topRate, double otherRate) {
        topRate_ = topRate;
        otherRate_ = otherRate;
    }
    
    /** 获取推荐的并行阈值 */
    static size_t getParallelThreshold() { return 5000; }
    
    /** 检查是否应该使用并行采样 */
    static bool shouldUseParallel(size_t sampleCount) {
        return sampleCount >= getParallelThreshold();
    }
    
    /** 估算采样内存使用 */
    static size_t estimateMemoryUsage(size_t totalSamples, double samplingRatio) {
        size_t selectedSamples = static_cast<size_t>(totalSamples * samplingRatio);
        return totalSamples * sizeof(std::pair<double, int>) + // 排序数组
               selectedSamples * (sizeof(int) + sizeof(double)); // 输出数组
    }

private:
    double topRate_;      // 大梯度保留比例
    double otherRate_;    // 小梯度采样比例  
    mutable std::mt19937 gen_;
    
    // =============================================
    // 内部并行优化方法
    // =============================================
    
    /** 大数据集的并行GOSS采样 */
    void sampleParallel(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;
    
    /** 小数据集的优化串行GOSS采样 */
    void sampleSerial(const std::vector<double>& gradients,
                      std::vector<int>& sampleIndices,
                      std::vector<double>& sampleWeights) const;
    
    // =============================================
    // 工具方法
    // =============================================
    
    /** 验证采样参数 */
    bool validateParameters() const {
        return topRate_ > 0.0 && topRate_ < 1.0 && 
               otherRate_ > 0.0 && otherRate_ < 1.0 &&
               (topRate_ + otherRate_) <= 1.0;
    }
    
    /** 计算理论采样率 */
    double getTheoreticalSamplingRatio() const {
        return topRate_ + (1.0 - topRate_) * otherRate_;
    }
};

#endif // LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP