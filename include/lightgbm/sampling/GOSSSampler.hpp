#ifndef LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP
#define LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP

#include <vector>
#include <random>

/** GOSS(Gradient-based One-Side Sampling)采样器 */
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
                
private:
    double topRate_;      // 大梯度保留比例
    double otherRate_;    // 小梯度采样比例  
    mutable std::mt19937 gen_;
};

#endif // LIGHTGBM_SAMPLING_GOSSSAMPLER_HPP
