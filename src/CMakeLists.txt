
# src/CMakeLists.txt
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# 各模块子目录
add_subdirectory(preprocessing)
add_subdirectory(functions/io)
add_subdirectory(pipeline)
add_subdirectory(tree)
add_subdirectory(boosting)  # 新增回归Boosting模块
add_subdirectory(app)

# 原有可执行文件
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

# 新的回归Boosting可执行文件
add_executable(RegressionBoostingMain
    regression_boosting_main.cpp
)

target_link_libraries(RegressionBoostingMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
)
# 添加XGBoost模块
add_subdirectory(xgboost)

# 添加XGBoost可执行文件
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