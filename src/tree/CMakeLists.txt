# 决策树核心模块，包含 criterion、finder、pruner、trainer
add_library(DecisionTree_lib
    criterion/MSECriterion.cpp
    criterion/MAECriterion.cpp
    criterion/HuberCriterion.cpp
    criterion/QuantileCriterion.cpp
    criterion/LogCoshCriterion.cpp
    criterion/PoissonCriterion.cpp
    finder/ExhaustiveSplitFinder.cpp
    finder/RandomSplitFinder.cpp
    finder/QuartileSplitFinder.cpp
    finder/HistogramEWFinder.cpp
    finder/HistogramEQFinder.cpp
    finder/AdaptiveEWFinder.cpp
    finder/AdaptiveEQFinder.cpp
    pruner/NoPruner.cpp
    trainer/SingleTreeTrainer.cpp
)

target_include_directories(DecisionTree_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 如果未来 SplitFinder 或 Criterion 有依赖，也在这里 link
