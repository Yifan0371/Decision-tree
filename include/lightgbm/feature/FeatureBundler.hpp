#ifndef LIGHTGBM_FEATURE_FEATUREBUNDLER_HPP
#define LIGHTGBM_FEATURE_FEATUREBUNDLER_HPP

#include <vector>
#include <unordered_set>

/** 特征绑定信息 */
struct FeatureBundle {
    std::vector<int> features;        // 绑定的特征索引
    std::vector<double> offsets;      // 特征偏移量
    int totalBins;                    // 绑定后总分箱数
};

/** EFB(Exclusive Feature Bundling)特征绑定器 */
class FeatureBundler {
public:
    explicit FeatureBundler(int maxBin = 255, double maxConflictRate = 0.0)
        : maxBin_(maxBin), maxConflictRate_(maxConflictRate) {}
    
    /**
     * 分析特征并创建绑定
     * @param data 训练数据
     * @param rowLength 特征数
     * @param sampleSize 样本数
     * @param bundles 输出：特征绑定信息
     */
    void createBundles(const std::vector<double>& data,
                      int rowLength, 
                      size_t sampleSize,
                      std::vector<FeatureBundle>& bundles) const;
    
    /**
     * 将原始特征值转换为绑定后的值
     * @param originalFeature 原始特征索引
     * @param value 原始特征值  
     * @param bundles 特征绑定信息
     * @return 绑定后的特征索引和值
     */
    std::pair<int, double> transformFeature(int originalFeature, 
                                           double value,
                                           const std::vector<FeatureBundle>& bundles) const;

private:
    int maxBin_;
    double maxConflictRate_;
    
    // 计算特征冲突率
    double calculateConflictRate(const std::vector<double>& data,
                               int rowLength, size_t sampleSize,
                               int feat1, int feat2) const;
    
    // 构建特征冲突图
    void buildConflictGraph(const std::vector<double>& data,
                          int rowLength, size_t sampleSize,
                          std::vector<std::vector<double>>& conflictMatrix) const;
};

#endif // LIGHTGBM_FEATURE_FEATUREBUNDLER_HPP
