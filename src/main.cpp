#include <chrono>
#include "app/SingleTreeApp.hpp"
#include <iostream>
#include <algorithm>

int main(int argc, char** argv)
{
    /* 默认 + 命令行覆盖（同前） */
    ProgramOptions opt;
    opt.dataPath       = "../data/data_clean/cleaned_data.csv";
    opt.maxDepth       = 2;
    opt.minSamplesLeaf = 50;
    opt.criterion      = "mse";
    if (argc >= 2) opt.dataPath       = argv[1];
    if (argc >= 3) opt.maxDepth       = std::stoi(argv[2]);
    if (argc >= 4) opt.minSamplesLeaf = std::stoi(argv[3]);
    if (argc >= 5) opt.criterion      = argv[4];

    std::cout << "Running with " << opt.criterion << " …\n";

    /* -------- 计时 -------- */
    auto t0 = std::chrono::steady_clock::now();

    runSingleTreeApp(opt);

    auto t1 = std::chrono::steady_clock::now();
    double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    std::cout << "Total time: " << seconds << " s\n";
    return 0;
}
