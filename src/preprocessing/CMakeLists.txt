# src/preprocessing/CMakeLists.txt

add_library(preprocessing
    DataCleaner.cpp
)

# 这里要把顶层 include/ 目录加入搜索路径
target_include_directories(preprocessing PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
