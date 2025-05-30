#include "app/SingleTreeApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // 1. 设定默认参数
    ProgramOptions opts;
    opts.dataPath       = "../data/data_clean/cleaned_data.csv";
    opts.maxDepth       = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion      = "mse";
    opts.splitMethod    = "exhaustive";
    opts.prunerType     = "none";      // 默认不剪枝
    opts.prunerParam    = 0.01;        // 默认参数
    opts.valSplit       = 0.2;         // 验证集比例

    // 2. 参数解析
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.maxDepth = std::stoi(argv[2]);
    if (argc >= 4) opts.minSamplesLeaf = std::stoi(argv[3]);
    if (argc >= 5) opts.criterion = argv[4];
    if (argc >= 6) opts.splitMethod = argv[5];
    if (argc >= 7) opts.prunerType = argv[6];        // 新增：剪枝类型
    if (argc >= 8) opts.prunerParam = std::stod(argv[7]);  // 新增：剪枝参数
    if (argc >= 9) opts.valSplit = std::stod(argv[8]);     // 新增：验证集比例
    
    // 3. 输出参数
    std::cout << "Data: " << opts.dataPath << " | ";
    std::cout << "Depth: " << opts.maxDepth << " | ";
    std::cout << "MinLeaf: " << opts.minSamplesLeaf << " | ";
    std::cout << "Criterion: " << opts.criterion << " | ";
    std::cout << "Split: " << opts.splitMethod << " | ";
    std::cout << "Pruner: " << opts.prunerType;
    if (opts.prunerType != "none") {
        std::cout << "(" << opts.prunerParam << ")";
    }
    std::cout << std::endl;

    // 4. 运行
    runSingleTreeApp(opts);
    return 0;
}