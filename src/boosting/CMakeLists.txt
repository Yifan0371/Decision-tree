
# =============================================================================
# src/boosting/CMakeLists.txt (简化版本，确保编译通过)
# =============================================================================

# 回归Boosting核心库 - 简化版本
add_library(RegressionBoosting_lib
    # 损失函数
    loss/IRegressionLoss.cpp
    
    # 策略
    strategy/GradientRegressionStrategy.cpp
    
    # 训练器
    trainer/GBRTTrainer.cpp
    # DART策略 (新增)
    dart/UniformDartStrategy.cpp
    
    # 应用
    app/RegressionBoostingApp.cpp
)

target_include_directories(RegressionBoosting_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(RegressionBoosting_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)
