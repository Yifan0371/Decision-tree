cmake_minimum_required(VERSION 3.10)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# —— 全局编译选项 —— 
# -O3 优化等级最高
# -march=native 针对本机架构
# -Wall -Wextra 打开常见警告
add_compile_options(-O3 -march=native -Wall -Wextra)

# 把 include/ 加到头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 将 src 子目录也纳入构建
add_subdirectory(src)
