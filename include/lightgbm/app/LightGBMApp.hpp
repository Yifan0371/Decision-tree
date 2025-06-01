#ifndef LIGHTGBM_APP_LIGHTGBMAPP_HPP
#define LIGHTGBM_APP_LIGHTGBMAPP_HPP

#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include <string>
#include <memory>

/** LightGBM应用程序参数 */
struct LightGBMAppOptions {
    std::string dataPath = "../data/data_clean/cleaned_data.csv";
    std::string objective = "regression";
    
    // LightGBM参数
    int numIterations = 100;
    double learningRate = 0.1;
    int maxDepth = -1;
    int numLeaves = 31;
    int minDataInLeaf = 20;
    
    // GOSS参数
    double topRate = 0.2;
    double otherRate = 0.1;
    
    // EFB参数
    int maxBin = 255;
    double maxConflictRate = 0.0;
    bool enableFeatureBundling = true;
    bool enableGOSS = true;
    
    // 训练控制
    bool verbose = true;
    int earlyStoppingRounds = 0;
    double tolerance = 1e-7;
    double valSplit = 0.2;
    
    // 正则化
    double lambda = 0.0;
    double minSplitGain = 0.0;
};

/** 运行LightGBM训练和评估 */
void runLightGBMApp(const LightGBMAppOptions& options);

/** 创建LightGBM训练器工厂方法 */
std::unique_ptr<LightGBMTrainer> createLightGBMTrainer(const LightGBMAppOptions& options);

/** 解析命令行参数 */
LightGBMAppOptions parseLightGBMCommandLine(int argc, char** argv);

/** 输出模型摘要信息 */
void printLightGBMModelSummary(const LightGBMTrainer* trainer, const LightGBMAppOptions& options);

#endif // LIGHTGBM_APP_LIGHTGBMAPP_HPP
