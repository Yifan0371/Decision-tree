
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

    // 2. 参数解析
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.maxDepth = std::stoi(argv[2]);
    if (argc >= 4) opts.minSamplesLeaf = std::stoi(argv[3]);
    if (argc >= 5) opts.criterion = argv[4];
    if (argc >= 6) opts.splitMethod = argv[5];
    
    // 3. 输出参数（简化版）
    std::cout << "Data: " << opts.dataPath << " | ";
    std::cout << "Depth: " << opts.maxDepth << " | ";
    std::cout << "MinLeaf: " << opts.minSamplesLeaf << " | ";
    std::cout << "Criterion: " << opts.criterion << " | ";
    std::cout << "Split: " << opts.splitMethod << std::endl;

    // 4. 运行
    runSingleTreeApp(opts);
    return 0;
}