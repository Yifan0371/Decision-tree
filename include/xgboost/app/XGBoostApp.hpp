
// =============================================================================
// include/xgboost/app/XGBoostApp.hpp
// =============================================================================
#ifndef XGBOOST_APP_XGBOOSTAPP_HPP
#define XGBOOST_APP_XGBOOSTAPP_HPP

#include "xgboost/core/XGBoostConfig.hpp"
#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"

/** XGBoost应用程序参数 */
struct XGBoostAppOptions {
    std::string dataPath = "../data/data_clean/cleaned_data.csv";
    std::string objective = "reg:squarederror";
    
    // XGBoost参数
    int numRounds = 100;
    double eta = 0.3;
    int maxDepth = 6;
    int minChildWeight = 1;
    double lambda = 1.0;
    double gamma = 0.0;
    double subsample = 1.0;
    double colsampleByTree = 1.0;
    
    // 训练控制
    bool verbose = true;
    int earlyStoppingRounds = 0;
    double tolerance = 1e-7;
    double valSplit = 0.2;
    
    // 性能参数
    bool useApproxSplit = false;
    int maxBins = 256;
};

/** 运行XGBoost训练和评估 */
void runXGBoostApp(const XGBoostAppOptions& options);

/** 创建XGBoost训练器工厂方法 */
std::unique_ptr<XGBoostTrainer> createXGBoostTrainer(const XGBoostAppOptions& options);

/** 解析命令行参数 */
XGBoostAppOptions parseXGBoostCommandLine(int argc, char** argv);

/** 输出XGBoost模型摘要信息 */
void printXGBoostModelSummary(const XGBoostTrainer* trainer, const XGBoostAppOptions& options);

#endif // XGBOOST_APP_XGBOOSTAPP_HPP