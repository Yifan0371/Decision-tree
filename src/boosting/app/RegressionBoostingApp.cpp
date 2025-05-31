
// =============================================================================
// src/boosting/app/RegressionBoostingApp.cpp (简化版本)
// =============================================================================
#include "boosting/app/RegressionBoostingApp.hpp"
#include "boosting/strategy/GradientRegressionStrategy.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "boosting/loss/HuberLoss.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void runRegressionBoostingApp(const RegressionBoostingOptions& opts) {
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
    auto trainer = createRegressionBoostingTrainer(opts);
    
    // 训练模型
    if (opts.verbose) {
        std::cout << "\n=== Training GBRT ===" << std::endl;
    }
    
    auto trainStart = std::chrono::high_resolution_clock::now();
    trainer->train(dp.X_train, dp.rowLength, dp.y_train);
    auto trainEnd = std::chrono::high_resolution_clock::now();
    
    // 评估模型
    double trainLoss, trainMSE, trainMAE;
    trainer->evaluate(dp.X_train, dp.rowLength, dp.y_train, trainLoss, trainMSE, trainMAE);
    
    double testLoss, testMSE, testMAE;
    trainer->evaluate(dp.X_test, dp.rowLength, dp.y_test, testLoss, testMSE, testMAE);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // 输出结果
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Algorithm: GBRT" << std::endl;
    std::cout << "Trees: " << trainer->getModel()->getTreeCount() << std::endl;
    std::cout << "Train Loss: " << std::fixed << std::setprecision(6) << trainLoss 
              << " | Train MSE: " << trainMSE << std::endl;
    std::cout << "Test Loss: " << testLoss 
              << " | Test MSE: " << testMSE << std::endl;
    std::cout << "Train Time: " << trainTime.count() << "ms" << std::endl;
}

std::unique_ptr<GBRTTrainer> createRegressionBoostingTrainer(const RegressionBoostingOptions& opts) {
    // 创建损失函数
    std::unique_ptr<IRegressionLoss> lossFunc;
    if (opts.lossFunction == "huber") {
        lossFunc = std::make_unique<HuberLoss>(opts.huberDelta);
    } else {
        lossFunc = std::make_unique<SquaredLoss>();
    }
    
    // 创建策略
    auto strategy = std::make_unique<GradientRegressionStrategy>(
        std::move(lossFunc), opts.learningRate, opts.useLineSearch);
    
    // 创建配置
    GBRTConfig config;
    config.numIterations = opts.numIterations;
    config.learningRate = opts.learningRate;
    config.maxDepth = opts.maxDepth;
    config.minSamplesLeaf = opts.minSamplesLeaf;
    config.verbose = opts.verbose;
    
    return std::make_unique<GBRTTrainer>(config, std::move(strategy));
}

RegressionBoostingOptions parseRegressionCommandLine(int argc, char** argv) {
    RegressionBoostingOptions opts;
    opts.dataPath = "../data/data_clean/cleaned_data.csv";
    
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.lossFunction = argv[2];
    if (argc >= 4) opts.numIterations = std::stoi(argv[3]);
    if (argc >= 5) opts.learningRate = std::stod(argv[4]);
    if (argc >= 6) opts.maxDepth = std::stoi(argv[5]);
    
    return opts;
}

void printRegressionModelSummary(const GBRTTrainer* trainer, const RegressionBoostingOptions& opts) {
    std::cout << "Loss Function: " << opts.lossFunction << std::endl;
    const auto& losses = trainer->getTrainingLoss();
    if (!losses.empty()) {
        std::cout << "Final Loss: " << losses.back() << std::endl;
    }
}
