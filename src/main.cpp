#include "app/SingleTreeApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // 1. 设定默认参数
    ProgramOptions opts;
    opts.dataPath       = "../data/data_clean/cleaned_data.csv";
    opts.maxDepth       = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion      = "mse";  // 新增：默认使用 MSE

    // 2. 如果用户提供了参数，就覆盖默认
    if (argc >= 2) {
        opts.dataPath = argv[1];
    }
    if (argc >= 3) {
        opts.maxDepth = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        opts.minSamplesLeaf = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        opts.criterion = argv[4];  // 新增：可以指定准则
    }
    
    // 3. 输出使用的参数
    std::cout << "=== Program Parameters ===" << std::endl;
    std::cout << "Data path: " << opts.dataPath << std::endl;
    std::cout << "Max depth: " << opts.maxDepth << std::endl;
    std::cout << "Min samples leaf: " << opts.minSamplesLeaf << std::endl;
    std::cout << "Criterion: " << opts.criterion << std::endl;
    std::cout << "===========================" << std::endl;

    // 4. 输出使用说明
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] 
                  << " <data_path> [max_depth] [min_samples_leaf] [criterion]" << std::endl;
        std::cout << "Available criteria: mse, mae, huber, quantile[:tau], logcosh, poisson" << std::endl;
        std::cout << "Example: " << argv[0] 
                  << " data.csv 10 5 quantile:0.9" << std::endl;
        std::cout << "Using default parameters...\n" << std::endl;
    }

    // 5. 运行
    runSingleTreeApp(opts);
    return 0;
}