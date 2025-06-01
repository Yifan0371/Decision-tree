#include "lightgbm/app/LightGBMApp.hpp"
#include <iostream>
#include <iomanip>
#include <string>

void printBanner() {
    std::cout << "===============================================" << std::endl;
    std::cout << "           LightGBM Implementation            " << std::endl;
    std::cout << "      Light Gradient Boosting Machine         " << std::endl;
    std::cout << "===============================================" << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "\nUSAGE:" << std::endl;
    std::cout << "  " << programName << " [OPTIONS]" << std::endl;
    
    std::cout << "\nREQUIRED:" << std::endl;
    std::cout << "  --data PATH           Training data CSV file path" << std::endl;
    
    std::cout << "\nMODEL PARAMETERS:" << std::endl;
    std::cout << "  --objective STR       Objective (default: regression)" << std::endl;
    std::cout << "  --num-iterations INT  Boosting rounds (default: 100)" << std::endl;
    std::cout << "  --learning-rate FLOAT Learning rate (default: 0.1)" << std::endl;
    std::cout << "  --num-leaves INT      Max leaves (default: 31)" << std::endl;
    std::cout << "  --max-depth INT       Max depth (-1=unlimited, default: -1)" << std::endl;
    std::cout << "  --min-data-in-leaf INT Min samples per leaf (default: 20)" << std::endl;
    
    std::cout << "\nGOSS PARAMETERS:" << std::endl;
    std::cout << "  --top-rate FLOAT      Large gradient retain ratio (default: 0.2)" << std::endl;
    std::cout << "  --other-rate FLOAT    Small gradient sample ratio (default: 0.1)" << std::endl;
    std::cout << "  --enable-goss         Enable GOSS sampling (default: true)" << std::endl;
    
    std::cout << "\nEFB PARAMETERS:" << std::endl;
    std::cout << "  --max-bin INT         Max histogram bins (default: 255)" << std::endl;
    std::cout << "  --max-conflict FLOAT  Max feature conflict rate (default: 0.0)" << std::endl;
    std::cout << "  --enable-bundling     Enable feature bundling (default: true)" << std::endl;
    
    std::cout << "\nEXAMPLES:" << std::endl;
    std::cout << "  Basic: " << programName << " --data data.csv" << std::endl;
    std::cout << "  Custom: " << programName << " --data data.csv --num-leaves 63 --learning-rate 0.05" << std::endl;
}

bool parseArguments(int argc, char** argv, LightGBMAppOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") return false;
        else if (arg == "--data" && i + 1 < argc) opts.dataPath = argv[++i];
        else if (arg == "--objective" && i + 1 < argc) opts.objective = argv[++i];
        else if (arg == "--num-iterations" && i + 1 < argc) opts.numIterations = std::stoi(argv[++i]);
        else if (arg == "--learning-rate" && i + 1 < argc) opts.learningRate = std::stod(argv[++i]);
        else if (arg == "--num-leaves" && i + 1 < argc) opts.numLeaves = std::stoi(argv[++i]);
        else if (arg == "--max-depth" && i + 1 < argc) opts.maxDepth = std::stoi(argv[++i]);
        else if (arg == "--min-data-in-leaf" && i + 1 < argc) opts.minDataInLeaf = std::stoi(argv[++i]);
        else if (arg == "--top-rate" && i + 1 < argc) opts.topRate = std::stod(argv[++i]);
        else if (arg == "--other-rate" && i + 1 < argc) opts.otherRate = std::stod(argv[++i]);
        else if (arg == "--max-bin" && i + 1 < argc) opts.maxBin = std::stoi(argv[++i]);
        else if (arg == "--max-conflict" && i + 1 < argc) opts.maxConflictRate = std::stod(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) opts.lambda = std::stod(argv[++i]);
        else if (arg == "--min-split-gain" && i + 1 < argc) opts.minSplitGain = std::stod(argv[++i]);
        else if (arg == "--enable-goss") opts.enableGOSS = true;
        else if (arg == "--disable-goss") opts.enableGOSS = false;
        else if (arg == "--enable-bundling") opts.enableFeatureBundling = true;
        else if (arg == "--disable-bundling") opts.enableFeatureBundling = false;
        else if (arg == "--verbose") opts.verbose = true;
        else if (arg == "--quiet") opts.verbose = false;
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return !opts.dataPath.empty();
}

int main(int argc, char** argv) {
    printBanner();
    
    // 默认参数
    LightGBMAppOptions opts;
    opts.dataPath = "";
    opts.objective = "regression";
    opts.numIterations = 100;
    opts.learningRate = 0.1;
    opts.maxDepth = -1;
    opts.numLeaves = 31;
    opts.minDataInLeaf = 20;
    opts.topRate = 0.2;
    opts.otherRate = 0.1;
    opts.maxBin = 255;
    opts.maxConflictRate = 0.0;
    opts.enableGOSS = true;
    opts.enableFeatureBundling = true;
    opts.verbose = true;
    opts.lambda = 0.0;
    opts.minSplitGain = 0.0;
    
    if (!parseArguments(argc, argv, opts)) {
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        if (opts.verbose) {
            std::cout << "Configuration:" << std::endl;
            std::cout << "Data: " << opts.dataPath << std::endl;
            std::cout << "Iterations: " << opts.numIterations << std::endl;
            std::cout << "Learning Rate: " << opts.learningRate << std::endl;
            std::cout << "Num Leaves: " << opts.numLeaves << std::endl;
            std::cout << "GOSS: " << (opts.enableGOSS ? "Yes" : "No") << std::endl;
            std::cout << "Feature Bundling: " << (opts.enableFeatureBundling ? "Yes" : "No") << std::endl;
            std::cout << std::endl;
        }
        
        runLightGBMApp(opts);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
