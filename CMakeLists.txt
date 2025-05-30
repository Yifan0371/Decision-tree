# 根目录 CMakeLists.txt - 添加性能优化配置

cmake_minimum_required(VERSION 3.16)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 性能优化编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # 基础优化
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    
    # 启用向量化和现代 CPU 指令集
    set(OPTIMIZATION_FLAGS 
        "-march=native"           # 针对本机 CPU 优化
        "-mtune=native"           # 针对本机 CPU 调优
        "-mavx2"                  # 启用 AVX2 指令集
        "-mfma"                   # 启用 FMA 指令
        "-ftree-vectorize"        # 启用自动向量化
        "-ffast-math"             # 快速数学运算（小心精度）
        "-funroll-loops"          # 循环展开
        "-fprefetch-loop-arrays"  # 数组预取
        "-flto"                   # 链接时优化
    )
    
    # 向量化报告（调试用）
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND OPTIMIZATION_FLAGS 
            "-fopt-info-vec"
            "-fopt-info-vec-missed"
        )
    endif()
    
    # 应用优化标志
    string(JOIN " " OPTIMIZATION_FLAGS_STR ${OPTIMIZATION_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS_STR}")
    
    # 链接时优化
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
endif()

# Intel 编译器优化
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost -ipo -qopt-report=5")
endif()

# 定义性能宏
add_compile_definitions(
    ENABLE_SIMD=1
    PERFORMANCE_MODE=1
)

# 包含目录
include_directories(include)

# 子目录
add_subdirectory(src)

# 添加性能测试目标
add_custom_target(perf_test
    COMMAND echo "Running performance optimized build..."
    COMMAND ${CMAKE_BINARY_DIR}/DecisionTreeMain
    DEPENDS DecisionTreeMain
)