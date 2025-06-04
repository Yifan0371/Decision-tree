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
