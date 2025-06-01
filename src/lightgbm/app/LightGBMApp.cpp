#include "lightgbm/app/LightGBMApp.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void runLightGBMApp(const LightGBMAppOptions& opts) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 读取数据
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);
    
    if (opts.verbose) {
        std::cout << "Loaded data: " << y.size() << " samples, " 
                  << (rowLength - 1) << " features" << std::endl;
    }
    
    // 划分数据集
    DataParams dp;
    splitDataset(X, y, rowLength, dp);
    
    // 创建训练器
    auto trainer = createLightGBMTrainer(opts);
    
    // 训练模型
    if (opts.verbose) {
        std::cout << "\n=== Training LightGBM ===" << std::endl;
    }
    
    auto trainStart = std::chrono::high_resolution_clock::now();
    trainer->train(dp.X_train, dp.rowLength, dp.y_train);
    auto trainEnd = std::chrono::high_resolution_clock::now();
    
    // 评估模型
    double trainMSE, trainMAE, testMSE, testMAE;
    trainer->evaluate(dp.X_train, dp.rowLength, dp.y_train, trainMSE, trainMAE);
    trainer->evaluate(dp.X_test, dp.rowLength, dp.y_test, testMSE, testMAE);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // 输出结果
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    std::cout << "\n=== LightGBM Results ===" << std::endl;
    std::cout << "Trees: " << trainer->getLGBModel()->getTreeCount() << std::endl;
    std::cout << "Train MSE: " << std::fixed << std::setprecision(6) << trainMSE 
              << " | Train MAE: " << trainMAE << std::endl;
    std::cout << "Test MSE: " << testMSE 
              << " | Test MAE: " << testMAE << std::endl;
    std::cout << "Train Time: " << trainTime.count() << "ms"
              << " | Total Time: " << totalTime.count() << "ms" << std::endl;
    
    printLightGBMModelSummary(trainer.get(), opts);
}

std::unique_ptr<LightGBMTrainer> createLightGBMTrainer(const LightGBMAppOptions& opts) {
    LightGBMConfig config;
    config.numIterations = opts.numIterations;
    config.learningRate = opts.learningRate;
    config.maxDepth = opts.maxDepth;
    config.numLeaves = opts.numLeaves;
    config.minDataInLeaf = opts.minDataInLeaf;
    config.topRate = opts.topRate;
    config.otherRate = opts.otherRate;
    config.maxBin = opts.maxBin;
    config.maxConflictRate = opts.maxConflictRate;
    config.enableFeatureBundling = opts.enableFeatureBundling;
    config.enableGOSS = opts.enableGOSS;
    config.verbose = opts.verbose;
    config.earlyStoppingRounds = opts.earlyStoppingRounds;
    config.tolerance = opts.tolerance;
    config.lambda = opts.lambda;
    config.minSplitGain = opts.minSplitGain;
    
    return std::make_unique<LightGBMTrainer>(config);
}

LightGBMAppOptions parseLightGBMCommandLine(int argc, char** argv) {
    LightGBMAppOptions opts;
    
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.objective = argv[2];
    if (argc >= 4) opts.numIterations = std::stoi(argv[3]);
    if (argc >= 5) opts.learningRate = std::stod(argv[4]);
    if (argc >= 6) opts.numLeaves = std::stoi(argv[5]);
    if (argc >= 7) opts.topRate = std::stod(argv[6]);
    if (argc >= 8) opts.otherRate = std::stod(argv[7]);
    
    return opts;
}

void printLightGBMModelSummary(const LightGBMTrainer* trainer, const LightGBMAppOptions& opts) {
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "Algorithm: LightGBM" << std::endl;
    std::cout << "Objective: " << opts.objective << std::endl;
    std::cout << "Learning Rate: " << opts.learningRate << std::endl;
    std::cout << "Num Leaves: " << opts.numLeaves << std::endl;
    std::cout << "GOSS: " << (opts.enableGOSS ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Feature Bundling: " << (opts.enableFeatureBundling ? "Enabled" : "Disabled") << std::endl;
    
    const auto& losses = trainer->getTrainingLoss();
    if (!losses.empty()) {
        std::cout << "Final Training Loss: " << std::fixed << std::setprecision(6) 
                  << losses.back() << std::endl;
    }
}
