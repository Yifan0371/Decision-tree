#include "tree/trainer/SingleTreeTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include "app/SingleTreeApp.hpp"
#include "criterion/MSECriterion.hpp"
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
    auto criterion = std::make_unique<MSECriterion>();
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
