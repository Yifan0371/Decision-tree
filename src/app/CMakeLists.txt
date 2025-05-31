# src/app/CMakeLists.txt

# 原来的 SingleTreeApp 库
add_library(SingleTreeApp_lib
    SingleTreeApp.cpp
)
target_include_directories(SingleTreeApp_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(SingleTreeApp_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)

# 新增的 BaggingApp 库
add_library(BaggingApp_lib
    BaggingApp.cpp
)
target_include_directories(BaggingApp_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(BaggingApp_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)

# —— 数据清洗可执行 —— 
add_executable(DataCleanApp
    DataCleanApp.cpp    # 你的清洗 main 函数文件
)
target_include_directories(DataCleanApp PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(DataCleanApp PRIVATE
    DataCleaner_lib
)

# —— 新增Bagging可执行文件 —— 
add_executable(BaggingMain
    BaggingMain.cpp
)
target_include_directories(BaggingMain PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(BaggingMain PRIVATE
    BaggingApp_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)