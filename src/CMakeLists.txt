# src/CMakeLists.txt

# 将所有可执行文件输出到 build/ 根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 各模块子目录
add_subdirectory(preprocessing)
add_subdirectory(functions/io)
add_subdirectory(pipeline)
add_subdirectory(tree)
add_subdirectory(boosting)  # 回归Boosting模块
add_subdirectory(xgboost)   # XGBoost 模块
add_subdirectory(lightgbm)  # LightGBM 模块
add_subdirectory(app)

# =================================================================
# 原有可执行文件
# =================================================================

add_executable(DecisionTreeMain
    main.cpp
)
target_link_libraries(DecisionTreeMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    SingleTreeApp_lib
    BaggingApp_lib
)

# =================================================================
# 回归Boosting可执行文件
# =================================================================

add_executable(RegressionBoostingMain
    regression_boosting_main.cpp
)
target_link_libraries(RegressionBoostingMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
)

# =================================================================
# XGBoost可执行文件
# =================================================================

add_executable(XGBoostMain
    xgboost_main.cpp
)
target_link_libraries(XGBoostMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
    XGBoost_lib
)

# =================================================================
# LightGBM可执行文件
# =================================================================

add_executable(LightGBMMain
    lightgbm_main.cpp
)
target_link_libraries(LightGBMMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
    LightGBM_lib
)
