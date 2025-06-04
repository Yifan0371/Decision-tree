#include "app/BaggingApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // 1. 设定默认参数
    BaggingOptions opts;
    opts.dataPath       = "../data/data_clean/cleaned_data.csv";
    opts.numTrees       = 10;          // 默认10棵树
    opts.sampleRatio    = 1.0;         // 默认100%采样率
    opts.maxDepth       = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion      = "mse";
    opts.splitMethod    = "exhaustive";
    opts.prunerType     = "none";
    opts.prunerParam    = 0.01;
    opts.seed           = 42;

    // 2. 参数解析
    if (argc >= 2)  opts.dataPath = argv[1];
    if (argc >= 3)  opts.numTrees = std::stoi(argv[2]);
    if (argc >= 4)  opts.sampleRatio = std::stod(argv[3]);
    if (argc >= 5)  opts.maxDepth = std::stoi(argv[4]);
    if (argc >= 6)  opts.minSamplesLeaf = std::stoi(argv[5]);
    if (argc >= 7)  opts.criterion = argv[6];
    if (argc >= 8)  opts.splitMethod = argv[7];
    if (argc >= 9)  opts.prunerType = argv[8];
    if (argc >= 10) opts.prunerParam = std::stod(argv[9]);
    if (argc >= 11) opts.seed = static_cast<uint32_t>(std::stoi(argv[10]));
    
    // 3. 输出参数
    std::cout << "=== Bagging Parameters ===" << std::endl;
    std::cout << "Data: " << opts.dataPath << std::endl;
    std::cout << "Trees: " << opts.numTrees << " | Sample Ratio: " << opts.sampleRatio << std::endl;
    std::cout << "Depth: " << opts.maxDepth << " | MinLeaf: " << opts.minSamplesLeaf << std::endl;
    std::cout << "Criterion: " << opts.criterion << " | Split: " << opts.splitMethod << std::endl;
    std::cout << "Pruner: " << opts.prunerType;
    if (opts.prunerType != "none") {
        std::cout << "(" << opts.prunerParam << ")";
    }
    std::cout << " | Seed: " << opts.seed << std::endl;

    // 4. 运行
    runBaggingApp(opts);
    return 0;
}