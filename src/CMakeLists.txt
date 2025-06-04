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
