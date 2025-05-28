cmake_minimum_required(VERSION 3.10)
project(DecisionTree LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# 全局优化和警告选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# 将所有可执行文件输出到 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 子模块
add_subdirectory(src/preprocessing)
add_subdirectory(src/app)
