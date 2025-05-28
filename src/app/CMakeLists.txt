# src/app/CMakeLists.txt

add_executable(DataCleanApp
    DataCleanApp.cpp
)

# 只链接 preprocessing 库，去掉 data 和 utils
target_link_libraries(DataCleanApp
    PRIVATE preprocessing
)
