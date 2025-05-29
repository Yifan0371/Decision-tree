# 决策树核心模块，包含 criterion、finder、pruner、trainer
add_library(DecisionTree_lib
    criterion/MSECriterion.cpp
    finder/ExhaustiveSplitFinder.cpp
    pruner/NoPruner.cpp
    trainer/SingleTreeTrainer.cpp
)

target_include_directories(DecisionTree_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 如果未来 SplitFinder 或 Criterion 有依赖，也在这里 link
