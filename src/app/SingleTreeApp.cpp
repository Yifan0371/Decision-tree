
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include "app/SingleTreeApp.hpp"

// 准则
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "criterion/HuberCriterion.hpp"
#include "criterion/QuantileCriterion.hpp"
#include "criterion/LogCoshCriterion.hpp"
#include "criterion/PoissonCriterion.hpp"

// 分割器
#include "finder/ExhaustiveSplitFinder.hpp"
#include "finder/RandomSplitFinder.hpp"
#include "finder/QuartileSplitFinder.hpp"
#include "finder/HistogramEWFinder.hpp"
#include "finder/HistogramEQFinder.hpp"
#include "finder/AdaptiveEWFinder.hpp"
#include "finder/AdaptiveEQFinder.hpp"

#include "pruner/NoPruner.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

std::unique_ptr<ISplitFinder> createSplitFinder(const std::string& method) {
    if (method == "exhaustive" || method == "exact") {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
    else if (method == "random" || method.find("random:") == 0) {
        int k = 10;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            k = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<RandomSplitFinder>(k);
    }
    else if (method == "quartile") {
        return std::make_unique<QuartileSplitFinder>();
    }
    else if (method == "histogram_ew" || method.find("histogram_ew:") == 0) {
        int bins = 64;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEWFinder>(bins);
    }
    else if (method == "histogram_eq" || method.find("histogram_eq:") == 0) {
        int bins = 64;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEQFinder>(bins);
    }
    else if (method == "adaptive_ew" || method.find("adaptive_ew:") == 0) {
        std::string rule = "sturges";
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            rule = method.substr(pos + 1);
        }
        return std::make_unique<AdaptiveEWFinder>(8, 128, rule);
    }
    else if (method == "adaptive_eq") {
        return std::make_unique<AdaptiveEQFinder>(5, 64, 0.1);
    }
    else {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
}

void runSingleTreeApp(const ProgramOptions& opts) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 1. 读 CSV
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);

    // 2. 划分数据
    DataParams dp;
    splitDataset(X, y, rowLength, dp);

    // 3. 构造分割器
    auto finder = createSplitFinder(opts.splitMethod);
    
    // 4. 构造准则
    std::unique_ptr<ISplitCriterion> criterion;
    const std::string crit = opts.criterion;

    if (crit == "mae")
        criterion = std::make_unique<MAECriterion>();
    else if (crit == "huber")
        criterion = std::make_unique<HuberCriterion>();
    else if (crit.rfind("quantile", 0) == 0) {
        double tau = 0.5;
        auto pos = crit.find(':');
        if (pos != std::string::npos)
            tau = std::stod(crit.substr(pos + 1));
        criterion = std::make_unique<QuantileCriterion>(tau);
    }
    else if (crit == "logcosh")
        criterion = std::make_unique<LogCoshCriterion>();
    else if (crit == "poisson")
        criterion = std::make_unique<PoissonCriterion>();
    else
        criterion = std::make_unique<MSECriterion>();

    auto pruner = std::make_unique<NoPruner>();

    SingleTreeTrainer trainer(std::move(finder),
                              std::move(criterion),
                              std::move(pruner),
                              opts.maxDepth,
                              opts.minSamplesLeaf);

    // 5. 训练（测量时间）
    auto trainStart = std::chrono::high_resolution_clock::now();
    trainer.train(dp.X_train, dp.rowLength, dp.y_train);
    auto trainEnd = std::chrono::high_resolution_clock::now();

    // 6. 评估
    double mse, mae;
    trainer.evaluate(dp.X_test, dp.rowLength, dp.y_test, mse, mae);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // 7. 计算时间
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);

    // 8. 输出结果
    std::cout << "MSE: " << std::fixed << std::setprecision(6) << mse 
              << " | MAE: " << mae 
              << " | Train: " << trainTime.count() << "ms"
              << " | Total: " << totalTime.count() << "ms" << std::endl;
}