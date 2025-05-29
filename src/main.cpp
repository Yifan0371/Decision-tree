#include "app/SingleTreeApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // 1. 设定默认参数
    ProgramOptions opts;
    opts.dataPath       = "../data/data_clean/cleaned_data.csv";
    opts.maxDepth       = 100;
    opts.minSamplesLeaf = 2;

    // 2. 如果用户提供了参数，就覆盖默认
    if (argc >= 4) {
        opts.dataPath       = argv[1];
        opts.maxDepth       = std::stoi(argv[2]);
        opts.minSamplesLeaf = std::stoi(argv[3]);
    } else {
        std::cout << "No command-line args, using defaults:\n"
                  << "  dataPath="       << opts.dataPath       << "\n"
                  << "  maxDepth="       << opts.maxDepth       << "\n"
                  << "  minSamplesLeaf=" << opts.minSamplesLeaf << "\n";
    }

    // 3. 运行
    runSingleTreeApp(opts);
    return 0;
}
