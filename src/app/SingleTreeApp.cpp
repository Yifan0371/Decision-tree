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

// 剪枝器
#include "pruner/NoPruner.hpp"
#include "pruner/MinGainPrePruner.hpp"
#include "pruner/CostComplexityPruner.hpp"
#include "pruner/ReducedErrorPruner.hpp"

#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

// 数据分割结构（扩展支持验证集）
struct ExtendedDataParams {
    std::vector<double> X_train;
    std::vector<double> y_train;
    std::vector<double> X_val;     // 验证集
    std::vector<double> y_val;
    std::vector<double> X_test;
    std::vector<double> y_test;
    int rowLength;
};

bool splitDatasetWithValidation(const std::vector<double>& X,
                                const std::vector<double>& y,
                                int rowLength,
                                double valSplit,
                                ExtendedDataParams& out) {
    int feat = rowLength - 1;
    size_t totalRows = y.size();
    
    if (valSplit > 0) {
        // 三分：训练集、验证集、测试集
        size_t trainRows = static_cast<size_t>(totalRows * 0.7);
        size_t valRows = static_cast<size_t>(totalRows * valSplit);
        
        out.rowLength = feat;
        out.X_train.assign(X.begin(), X.begin() + trainRows * feat);
        out.y_train.assign(y.begin(), y.begin() + trainRows);
        
        out.X_val.assign(X.begin() + trainRows * feat, 
                        X.begin() + (trainRows + valRows) * feat);
        out.y_val.assign(y.begin() + trainRows, y.begin() + trainRows + valRows);
        
        out.X_test.assign(X.begin() + (trainRows + valRows) * feat, X.end());
        out.y_test.assign(y.begin() + trainRows + valRows, y.end());
    } else {
        // 二分：训练集、测试集
        size_t trainRows = static_cast<size_t>(totalRows * 0.8);
        
        out.rowLength = feat;
        out.X_train.assign(X.begin(), X.begin() + trainRows * feat);
        out.y_train.assign(y.begin(), y.begin() + trainRows);
        out.X_test.assign(X.begin() + trainRows * feat, X.end());
        out.y_test.assign(y.begin() + trainRows, y.end());
    }
    return true;
}

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

std::unique_ptr<IPruner> createPruner(const std::string& type, 
                                     double param,
                                     const std::vector<double>& X_val,
                                     int rowLength,
                                     const std::vector<double>& y_val) {
    if (type == "mingain") {
        return std::make_unique<MinGainPrePruner>(param);
    }
    else if (type == "cost_complexity") {
        return std::make_unique<CostComplexityPruner>(param);
    }
    else if (type == "reduced_error") {
        if (X_val.empty() || y_val.empty()) {
            std::cerr << "Warning: No validation data for reduced_error pruner, using NoPruner" << std::endl;
            return std::make_unique<NoPruner>();
        }
        return std::make_unique<ReducedErrorPruner>(X_val, rowLength, y_val);
    }
    else {
        return std::make_unique<NoPruner>();
    }
}

void runSingleTreeApp(const ProgramOptions& opts) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 1. 读 CSV
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);

    // 2. 划分数据（根据是否需要验证集）
    ExtendedDataParams dp;
    double valSplit = (opts.prunerType == "reduced_error") ? opts.valSplit : 0.0;
    splitDatasetWithValidation(X, y, rowLength, valSplit, dp);

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

    // 5. 构造剪枝器
    auto pruner = createPruner(opts.prunerType, opts.prunerParam, 
                              dp.X_val, dp.rowLength, dp.y_val);

    SingleTreeTrainer trainer(std::move(finder),
                              std::move(criterion),
                              std::move(pruner),
                              opts.maxDepth,
                              opts.minSamplesLeaf);

    // 6. 训练（测量时间）
    auto trainStart = std::chrono::high_resolution_clock::now();
    trainer.train(dp.X_train, dp.rowLength, dp.y_train);
    auto trainEnd = std::chrono::high_resolution_clock::now();

    // 7. 评估
    double mse, mae;
    trainer.evaluate(dp.X_test, dp.rowLength, dp.y_test, mse, mae);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // 8. 计算时间
    auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);

    // 9. 输出结果
    std::cout << " | Pruner: " << opts.prunerType;
    if (opts.prunerType != "none") {
        std::cout << "(" << opts.prunerParam << ")";
    }
    std::cout << std::endl;
    
    std::cout << "MSE: " << std::fixed << std::setprecision(6) << mse 
              << " | MAE: " << mae 
              << " | Train: " << trainTime.count() << "ms"
              << " | Total: " << totalTime.count() << "ms" << std::endl;
}