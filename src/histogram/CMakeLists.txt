# =============================================================================
# src/histogram/CMakeLists.txt - 预计算直方图优化模块（干净版本）
# =============================================================================

# 预计算直方图优化库
add_library(HistogramOptimized_lib
    PrecomputedHistograms.cpp
)

target_include_directories(HistogramOptimized_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# **OpenMP配置** - 避免重复定义
if(OpenMP_CXX_FOUND)
    target_link_libraries(HistogramOptimized_lib PUBLIC OpenMP::OpenMP_CXX)
    # 不手动定义 _OPENMP，让OpenMP::OpenMP_CXX自动处理
endif()

# 编译优化选项（移除verbose输出）
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HistogramOptimized_lib PRIVATE
        -O3                    # 最高优化级别
        -march=native          # 针对当前CPU优化
        -funroll-loops         # 循环展开
        -ffast-math           # 快速数学运算
        -ftree-vectorize      # 自动向量化
        # 移除verbose优化信息输出
        -fno-tree-loop-distribute-patterns  # 减少部分优化噪音
    )
endif()

# MSVC优化
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(HistogramOptimized_lib PRIVATE
        /O2                    # 最大优化
        /arch:AVX2            # AVX2指令集
        /fp:fast              # 快速浮点运算
    )
endif()

# 警告控制 - 减少不必要的警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HistogramOptimized_lib PRIVATE
        -Wall                  # 基本警告
        -Wno-unused-parameter  # 忽略未使用参数警告
        -Wno-unused-variable   # 忽略未使用变量警告
        -Wno-sign-compare      # 忽略符号比较警告
    )
endif()

# 性能分析支持（可选，默认关闭）
option(ENABLE_HISTOGRAM_PROFILING "Enable histogram performance profiling" OFF)
if(ENABLE_HISTOGRAM_PROFILING)
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_PROFILING)
endif()

# 内存使用优化选项
option(ENABLE_HISTOGRAM_MEMORY_OPTIMIZATION "Enable aggressive memory optimization" ON)
if(ENABLE_HISTOGRAM_MEMORY_OPTIMIZATION)
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_MEMORY_OPT)
endif()

# Debug模式特殊处理
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_DEBUG)
    # Debug模式下使用较低优化等级，便于调试
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(HistogramOptimized_lib PRIVATE -g -O1)
    endif()
endif()

# **链接时优化**（Release模式，静默执行）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(HistogramOptimized_lib PRIVATE -flto)
        target_link_options(HistogramOptimized_lib PRIVATE -flto)
    endif()
endif()

# 确保C++17或更高版本
target_compile_features(HistogramOptimized_lib PUBLIC cxx_std_17)