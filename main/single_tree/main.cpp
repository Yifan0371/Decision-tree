#include "app/SingleTreeApp.hpp"
#include "app/BaggingApp.hpp"
#include <iostream>
#include <string>
#include <cstdint>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [mode] [options...]" << std::endl;
    std::cout << "\nModes:" << std::endl;
    std::cout << "  single  - Single decision tree (default)" << std::endl;
    std::cout << "  bagging - Bootstrap aggregating" << std::endl;
    std::cout << "\nSingle Tree Options:" << std::endl;
    std::cout << "  " << programName << " single [dataPath] [maxDepth] [minSamplesLeaf] [criterion] [splitMethod] [prunerType] [prunerParam] [valSplit]" << std::endl;
    std::cout << "\nBagging Options:" << std::endl;
    std::cout << "  " << programName << " bagging [dataPath] [numTrees] [sampleRatio] [maxDepth] [minSamplesLeaf] [criterion] [splitMethod] [prunerType] [prunerParam] [seed]" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << programName << " single ../data/data_clean/cleaned_data.csv 10 2 mse exhaustive none" << std::endl;
    std::cout << "  " << programName << " bagging ../data/data_clean/cleaned_data.csv 50 1.0 10 2 mse random none" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode = argv[1];
    
    if (mode == "single") {
        // 单树模式
        ProgramOptions opts;
        opts.dataPath       = "../data/data_clean/cleaned_data.csv";
        opts.maxDepth       = 800;
        opts.minSamplesLeaf = 2;
        opts.criterion      = "mse";
        opts.splitMethod    = "exhaustive";
        opts.prunerType     = "none";
        opts.prunerParam    = 0.01;
        opts.valSplit       = 0.2;

        // 解析参数（从argv[2]开始）
        if (argc >= 3) opts.dataPath = argv[2];
        if (argc >= 4) opts.maxDepth = std::stoi(argv[3]);
        if (argc >= 5) opts.minSamplesLeaf = std::stoi(argv[4]);
        if (argc >= 6) opts.criterion = argv[5];
        if (argc >= 7) opts.splitMethod = argv[6];
        if (argc >= 8) opts.prunerType = argv[7];
        if (argc >= 9) opts.prunerParam = std::stod(argv[8]);
        if (argc >= 10) opts.valSplit = std::stod(argv[9]);
        
        std::cout << "=== Single Tree Mode ===" << std::endl;
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

        runSingleTreeApp(opts);
    }
    else if (mode == "bagging") {
        // Bagging模式
        BaggingOptions opts;
        opts.dataPath       = "../data/data_clean/cleaned_data.csv";
        opts.numTrees       = 10;
        opts.sampleRatio    = 1.0;
        opts.maxDepth       = 800;
        opts.minSamplesLeaf = 2;
        opts.criterion      = "mse";
        opts.splitMethod    = "exhaustive";
        opts.prunerType     = "none";
        opts.prunerParam    = 0.01;
        opts.seed           = 42;

        // 解析参数（从argv[2]开始）
        if (argc >= 3)  opts.dataPath = argv[2];
        if (argc >= 4)  opts.numTrees = std::stoi(argv[3]);
        if (argc >= 5)  opts.sampleRatio = std::stod(argv[4]);
        if (argc >= 6)  opts.maxDepth = std::stoi(argv[5]);
        if (argc >= 7)  opts.minSamplesLeaf = std::stoi(argv[6]);
        if (argc >= 8)  opts.criterion = argv[7];
        if (argc >= 9)  opts.splitMethod = argv[8];
        if (argc >= 10) opts.prunerType = argv[9];
        if (argc >= 11) opts.prunerParam = std::stod(argv[10]);
        if (argc >= 12) opts.seed = static_cast<uint32_t>(std::stoi(argv[11]));
        
        std::cout << "=== Bagging Mode ===" << std::endl;
        std::cout << "Data: " << opts.dataPath << std::endl;
        std::cout << "Trees: " << opts.numTrees << " | Sample Ratio: " << opts.sampleRatio << std::endl;
        std::cout << "Depth: " << opts.maxDepth << " | MinLeaf: " << opts.minSamplesLeaf << std::endl;
        std::cout << "Criterion: " << opts.criterion << " | Split: " << opts.splitMethod << std::endl;
        std::cout << "Pruner: " << opts.prunerType;
        if (opts.prunerType != "none") {
            std::cout << "(" << opts.prunerParam << ")";
        }
        std::cout << " | Seed: " << opts.seed << std::endl;

        runBaggingApp(opts);
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}