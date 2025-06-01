#include "xgboost/app/XGBoostApp.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void runXGBoostApp(const XGBoostAppOptions& opts) {
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
    auto trainer = createXGBoostTrainer(opts);
    
    // 设置验证集（如果需要早停）
    if (opts.earlyStoppingRounds > 0 && opts.valSplit > 0) {
        // 从训练集中再分出验证集
        size_t trainSize = dp.y_train.size();
        size_t valSize = static_cast<size_t>(trainSize * opts.valSplit);
        
        if (valSize > 0) {
            std::vector<double> X_val(dp.X_train.begin() + (trainSize - valSize) * dp.rowLength,
                                     dp.X_train.end());
            std::vector<double> y_val(dp.y_train.begin() + (trainSize - valSize),
                                     dp.y_train.end());
            
            // 调整训练集大小
            dp.X_train.resize((trainSize - valSize) * dp.rowLength);
            dp.y_train.resize(trainSize - valSize);
            
            trainer->setValidationData(X_val, y_val, dp.rowLength);
        }
    }
    
    // 训练模型
    if (opts.verbose) {
        std::cout << "\n=== Training XGBoost ===" << std::endl;
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
    
    std::cout << "\n=== XGBoost Results ===" << std::endl;
    std::cout << "Trees: " << trainer->getXGBModel()->getTreeCount() << std::endl;
    std::cout << "Train MSE: " << std::fixed << std::setprecision(6) << trainMSE 
              << " | Train MAE: " << trainMAE << std::endl;
    std::cout << "Test MSE: " << testMSE 
              << " | Test MAE: " << testMAE << std::endl;
    std::cout << "Train Time: " << trainTime.count() << "ms"
              << " | Total Time: " << totalTime.count() << "ms" << std::endl;
    
    // 输出特征重要性
    auto importance = trainer->getFeatureImportance(dp.rowLength);
    std::cout << "\nTop 10 Feature Importances:" << std::endl;
    std::vector<std::pair<double, int>> impWithIndex;
    for (int i = 0; i < static_cast<int>(importance.size()); ++i) {
        impWithIndex.emplace_back(importance[i], i);
    }
    
    std::sort(impWithIndex.begin(), impWithIndex.end(), std::greater<>());
    for (int i = 0; i < std::min(10, static_cast<int>(impWithIndex.size())); ++i) {
        std::cout << "Feature " << impWithIndex[i].second 
                  << ": " << std::fixed << std::setprecision(4) 
                  << impWithIndex[i].first << std::endl;
    }
    
    // 输出模型摘要
    printXGBoostModelSummary(trainer.get(), opts);
}

std::unique_ptr<XGBoostTrainer> createXGBoostTrainer(const XGBoostAppOptions& opts) {
    XGBoostConfig config;
    config.numRounds = opts.numRounds;
    config.eta = opts.eta;
    config.maxDepth = opts.maxDepth;
    config.minChildWeight = opts.minChildWeight;
    config.lambda = opts.lambda;
    config.gamma = opts.gamma;
    config.subsample = opts.subsample;
    config.colsampleByTree = opts.colsampleByTree;
    config.verbose = opts.verbose;
    config.earlyStoppingRounds = opts.earlyStoppingRounds;
    config.tolerance = opts.tolerance;
    config.useApproxSplit = opts.useApproxSplit;
    config.maxBins = opts.maxBins;
    config.objective = opts.objective;
    
    return std::make_unique<XGBoostTrainer>(config);
}

XGBoostAppOptions parseXGBoostCommandLine(int argc, char** argv) {
    XGBoostAppOptions opts;
    
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.objective = argv[2];
    if (argc >= 4) opts.numRounds = std::stoi(argv[3]);
    if (argc >= 5) opts.eta = std::stod(argv[4]);
    if (argc >= 6) opts.maxDepth = std::stoi(argv[5]);
    if (argc >= 7) opts.lambda = std::stod(argv[6]);
    if (argc >= 8) opts.gamma = std::stod(argv[7]);
    
    return opts;
}

void printXGBoostModelSummary(const XGBoostTrainer* trainer, const XGBoostAppOptions& opts) {
    std::cout << "\n=== Model Summary ===" << std::endl;
    std::cout << "Algorithm: XGBoost" << std::endl;
    std::cout << "Objective: " << opts.objective << std::endl;
    std::cout << "Learning Rate: " << opts.eta << std::endl;
    std::cout << "Max Depth: " << opts.maxDepth << std::endl;
    std::cout << "Lambda: " << opts.lambda << std::endl;
    std::cout << "Gamma: " << opts.gamma << std::endl;
    
    const auto& losses = trainer->getTrainingLoss();
    if (!losses.empty()) {
        std::cout << "Final Training Loss: " << std::fixed << std::setprecision(6) 
                  << losses.back() << std::endl;
    }
}