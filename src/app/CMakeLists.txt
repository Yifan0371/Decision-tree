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

# —— 新增数据清洗可执行 —— 
add_executable(DataCleanApp
    DataCleanApp.cpp    # 你的清洗 main 函数文件
)
target_include_directories(DataCleanApp PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(DataCleanApp PRIVATE
    DataCleaner_lib
    # 如果 DataCleanApp 里也用到了 filesystem，不用额外 link
)
