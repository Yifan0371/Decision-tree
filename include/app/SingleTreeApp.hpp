
#ifndef APP_SINGLE_TREE_APP_HPP
#define APP_SINGLE_TREE_APP_HPP
#include <string>

/** 运行参数 */
struct ProgramOptions {
    std::string dataPath;        // CSV 路径
    int         maxDepth;        // 树最大深度
    int         minSamplesLeaf;  // 叶子最小样本数
    std::string criterion;       // "mse" | "mae" | "huber" | "quantile[:τ]" | "logcosh" | "poisson"
    std::string splitMethod;     // 新增：分割方法选择
};

/** 训练 + 评估 单棵树 */
void runSingleTreeApp(const ProgramOptions&);

#endif // APP_SINGLE_TREE_APP_HPP