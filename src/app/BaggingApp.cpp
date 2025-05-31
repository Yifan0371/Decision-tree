#include "app/BaggingApp.hpp"
#include "ensemble/BaggingTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"

#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <algorithm>

void runBaggingApp(const BaggingOptions& opts) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 1. 读 CSV
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);

    // 2. 划分数据集 (80/20)
    DataParams dp;
    if (!splitDataset(X, y, rowLength, dp)) {
        std::cerr << "Failed to split dataset" << std::endl;
        return;
    }

    // 3. 创建Bagging训练器
    BaggingTrainer trainer(
        opts.numTrees,
        opts.sampleRatio,
        opts.maxDepth,
        opts.minSamplesLeaf,
        opts.criterion,
        opts.splitMethod,
        opts.prunerType,
        opts.prunerParam,
        opts.seed
    );

    // 4. 训练（测量时间）
    auto trainStart = std::chrono::high_resolution_clock::now();
    trainer.train(dp.X_train, dp.rowLength, dp.y_train);
    auto trainEnd = std::chrono::high_resolution_clock::now();

    // 5. 评估
    double mse, mae;
    trainer.evaluate(dp.X_test, dp.rowLength, dp.y_test, mse, mae);
    
    // 6. 计算OOB误差
    double oobError = trainer.getOOBError(dp.X_train, dp.rowLength, dp.y_train);
    
    // 7. 计算特征重要性
    auto featureImportance = trainer.getFeatureImportance(dp.rowLength);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // 8. 计算时间
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);

    // 9. 输出结果
    std::cout << "\n=== Bagging Results ===" << std::endl;
    std::cout << "Trees: " << opts.numTrees 
              << " | Sample Ratio: " << std::fixed << std::setprecision(2) << opts.sampleRatio
              << " | Criterion: " << opts.criterion 
              << " | Split: " << opts.splitMethod << std::endl;
    
    std::cout << "Test MSE: " << std::fixed << std::setprecision(6) << mse 
              << " | Test MAE: " << mae << std::endl;
    
    std::cout << "OOB MSE: " << std::fixed << std::setprecision(6) << oobError << std::endl;
    
    std::cout << "Train Time: " << trainTime.count() << "ms"
              << " | Total Time: " << totalTime.count() << "ms" << std::endl;
    
    // 10. 输出特征重要性（前10个最重要的特征）
    std::cout << "\nTop 10 Feature Importances:" << std::endl;
    std::vector<std::pair<double, int>> importanceWithIndex;
    for (int i = 0; i < static_cast<int>(featureImportance.size()); ++i) {
        importanceWithIndex.emplace_back(featureImportance[i], i);
    }
    
    std::sort(importanceWithIndex.begin(), importanceWithIndex.end(), 
              std::greater<std::pair<double, int>>());
    
    for (int i = 0; i < std::min(10, static_cast<int>(importanceWithIndex.size())); ++i) {
        std::cout << "Feature " << importanceWithIndex[i].second 
                  << ": " << std::fixed << std::setprecision(4) 
                  << importanceWithIndex[i].first << std::endl;
    }
}