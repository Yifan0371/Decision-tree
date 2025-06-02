
// =============================================================================
// include/boosting/app/RegressionBoostingApp.hpp
// =============================================================================
#ifndef BOOSTING_APP_REGRESSIONBOOSTINGAPP_HPP
#define BOOSTING_APP_REGRESSIONBOOSTINGAPP_HPP

#include "../trainer/GBRTTrainer.hpp"
#include <string>
#include <memory>

/** 回归Boosting应用程序参数 */
struct RegressionBoostingOptions {
    // 数据参数
    std::string dataPath;              
    std::string lossFunction = "squared"; // "squared", "huber", "absolute", "quantile"
    
    // GBRT参数
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    // 训练控制
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    double valSplit = 0.2;             
    
    // 损失函数参数
    double huberDelta = 1.0;           
    double quantile = 0.5;             
    
    // 性能参数
    bool useLineSearch = false;        
    double subsample = 1.0;    
     // === 新增DART参数 ===
    bool enableDart = false;
    double dartDropRate = 0.1;
    bool dartNormalize = true;
    bool dartSkipDropForPrediction = false;
    std::string dartStrategy = "uniform";
    uint32_t dartSeed = 42;        
};

/** 运行回归Boosting训练和评估 */
void runRegressionBoostingApp(const RegressionBoostingOptions& options);

/** 创建回归Boosting训练器工厂方法 */
std::unique_ptr<GBRTTrainer> createRegressionBoostingTrainer(const RegressionBoostingOptions& options);

/** 解析命令行参数 */
RegressionBoostingOptions parseRegressionCommandLine(int argc, char** argv);

/** 输出回归模型摘要信息 */
void printRegressionModelSummary(const GBRTTrainer* trainer, const RegressionBoostingOptions& options);

#endif // BOOSTING_APP_REGRESSIONBOOSTINGAPP_HPP