# =============================================================================
# src/xgboost/CMakeLists.txt
# =============================================================================

# XGBoost核心库
add_library(XGBoost_lib
    # 损失函数
    loss/XGBoostLossFactory.cpp
    
    # 分裂器
    finder/XGBoostSplitFinder.cpp
    
    # 训练器
    trainer/XGBoostTrainer.cpp
    
    # 应用
    app/XGBoostApp.cpp
)

target_include_directories(XGBoost_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(XGBoost_lib PUBLIC
    DecisionTree_lib
    RegressionBoosting_lib  # 复用现有的损失函数基础设施
    DataIO_lib
    DataSplit_lib
)
