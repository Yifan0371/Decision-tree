#include "tree/trainer/SingleTreeTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include "app/SingleTreeApp.hpp"
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "criterion/HuberCriterion.hpp"
#include "criterion/QuantileCriterion.hpp"
#include "criterion/LogCoshCriterion.hpp"
#include "criterion/PoissonCriterion.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"
#include "pruner/NoPruner.hpp"
#include <iostream>

void runSingleTreeApp(const ProgramOptions& opts) {
    // 1. 读 CSV
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);

    // 2. 划分数据
    DataParams dp;
    splitDataset(X, y, rowLength, dp);

    // 3. 构造策略并训练
    auto finder    = std::make_unique<ExhaustiveSplitFinder>();
    // 3. 构造准则
    std::unique_ptr<ISplitCriterion> criterion;
    const std::string crit = opts.criterion;

    if (crit == "mae")
        criterion = std::make_unique<MAECriterion>();

    else if (crit == "huber")
        criterion = std::make_unique<HuberCriterion>();               // 默认 δ=1.0

    else if (crit.rfind("quantile", 0) == 0) {                        // 形如 quantile:0.9
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

    else   // 默认 MSE
        criterion = std::make_unique<MSECriterion>();

    auto pruner    = std::make_unique<NoPruner>();

    SingleTreeTrainer trainer(std::move(finder),
                              std::move(criterion),
                              std::move(pruner),
                              opts.maxDepth,
                              opts.minSamplesLeaf);

    trainer.train(dp.X_train, dp.rowLength, dp.y_train);

    // 4. 评估
    double mse, mae;
    trainer.evaluate(dp.X_test, dp.rowLength, dp.y_test, mse, mae);

    std::cout << "MSE = " << mse << "\n";
    std::cout << "MAE = " << mae << "\n";
}
