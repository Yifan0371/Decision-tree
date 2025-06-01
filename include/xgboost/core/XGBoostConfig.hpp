// =============================================================================
// include/xgboost/core/XGBoostConfig.hpp
// =============================================================================
#ifndef XGBOOST_CORE_XGBOOSTCONFIG_HPP
#define XGBOOST_CORE_XGBOOSTCONFIG_HPP

#include <string> // 包含字符串处理
#include <vector> // 包含向量处理
#include <memory> // 包含智能指针
#include <tuple>  // 包含元组处理
#include <iostream> // 包含输入输出流           
/** XGBoost训练配置 */
struct XGBoostConfig {
    // 基础参数
    int numRounds = 100;              // boosting轮数
    double eta = 0.3;                 // 学习率（shrinkage）
    int maxDepth = 6;                 // 树最大深度
    int minChildWeight = 1;           // 最小子节点权重和
    
    // 正则化参数
    double lambda = 1.0;              // L2正则化参数
    double gamma = 0.0;               // 最小分裂损失
    double alpha = 0.0;               // L1正则化参数（未实现）
    
    // 采样参数
    double subsample = 1.0;           // 行采样比例
    double colsampleByTree = 1.0;     // 列采样比例
    
    // 训练控制
    bool verbose = true;              // 是否输出训练信息
    int earlyStoppingRounds = 0;      // 早停轮数
    double tolerance = 1e-7;          // 收敛容差
    
    // 性能优化
    bool useApproxSplit = false;      // 是否使用近似分裂算法
    int maxBins = 256;                // 直方图最大分箱数
    
    // 目标函数
    std::string objective = "reg:squarederror";  // 回归目标函数
};

#endif // XGBOOST_CORE_XGBOOSTCONFIG_HPP