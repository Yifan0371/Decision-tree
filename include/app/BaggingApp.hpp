#ifndef APP_BAGGING_APP_HPP
#define APP_BAGGING_APP_HPP
#include <string>
#include <cstdint>
/** Bagging运行参数 */
struct BaggingOptions {
    std::string dataPath;        // CSV 路径
    int         numTrees;        // 树的数量
    double      sampleRatio;     // Bootstrap采样比例
    int         maxDepth;        // 树最大深度
    int         minSamplesLeaf;  // 叶子最小样本数
    std::string criterion;       // "mse" | "mae" | "huber" | "quantile[:τ]" | "logcosh" | "poisson"
    std::string splitMethod;     // 分割方法选择
    std::string prunerType;      // "none" | "mingain" | "cost_complexity" | "reduced_error"
    double      prunerParam;     // 剪枝参数
    uint32_t    seed;           // 随机种子
};

/** 训练 + 评估 Bagging模型 */
void runBaggingApp(const BaggingOptions& opts);

#endif // APP_BAGGING_APP_HPP