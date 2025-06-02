#ifndef LIGHTGBM_CORE_LIGHTGBMCONFIG_HPP
#define LIGHTGBM_CORE_LIGHTGBMCONFIG_HPP

#include <string>

/** LightGBM训练配置 */
struct LightGBMConfig {
    // 基础参数
    int numIterations = 100;          // boosting轮数
    double learningRate = 0.1;        // 学习率
    int maxDepth = -1;                // 树最大深度 (-1表示无限制)
    int numLeaves = 31;               // 叶子节点数量限制
    int minDataInLeaf = 20;           // 叶节点最小数据量
    
    // GOSS参数
    double topRate = 0.2;             // 大梯度样本保留比例
    double otherRate = 0.1;           // 小梯度样本采样比例
    
    // EFB参数  
    int maxBin = 255;                 // 直方图最大分箱数
    double maxConflictRate = 0.0;     // 特征冲突率阈值
    
    // 训练控制
    bool verbose = true;              
    int earlyStoppingRounds = 0;      
    double tolerance = 1e-7;          
    
    // 正则化
    double lambda = 0.0;              // L2正则化
    double minSplitGain = 0.0;        // 最小分裂增益
    
    // 性能优化
    bool enableFeatureBundling = true; // 是否启用特征绑定
    bool enableGOSS = true;           // 是否启用GOSS采样
    int histPoolSize = 16384;         // 直方图缓冲池大小
    
    // 目标函数
    std::string objective = "regression"; // regression, binary, multiclass

    // **新增：分割器配置**
    std::string splitMethod = "histogram_ew";  // 分割方法选择
    int histogramBins = 255;           // 直方图分箱数
    std::string adaptiveRule = "sturges"; // 自适应规则
    int minSamplesPerBin = 5;          // 等频分箱最小样本数
    int maxAdaptiveBins = 128;         // 自适应最大分箱数
    double variabilityThreshold = 0.1; // 变异性阈值
    bool enableSIMD = true;            // SIMD优化
};

#endif // LIGHTGBM_CORE_LIGHTGBMCONFIG_HPP
