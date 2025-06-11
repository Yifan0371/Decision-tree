# =============================================================================
# src/tree/CMakeLists.txt - 决策树模块（干净版本）
# =============================================================================
# Add this section at the end of src/tree/CMakeLists.txt

# =============================================================================
# MPI-enabled Bagging (Optional)
# =============================================================================
if(ENABLE_MPI AND MPI_CXX_FOUND)
    # MPI Bagging library as STATIC library
    add_library(MPIBagging_lib STATIC
        ensemble/MPIBaggingTrainer.cpp
    )
    
    target_include_directories(MPIBagging_lib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )
    
    # Link dependencies
    target_link_libraries(MPIBagging_lib PUBLIC
        DecisionTree_lib          # Includes BaggingTrainer
        HistogramOptimized_lib
        MPI::MPI_CXX
    )
    
    # OpenMP support for hybrid parallelism
    if(OpenMP_CXX_FOUND)
        target_link_libraries(MPIBagging_lib PUBLIC OpenMP::OpenMP_CXX)
    endif()
    
    # Compiler flags for MPI
    target_compile_options(MPIBagging_lib PRIVATE
        ${MPI_CXX_COMPILE_FLAGS}
    )
    
    # Performance optimizations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(MPIBagging_lib PRIVATE
            -O3
            -march=native
            -funroll-loops
            -ftree-vectorize
        )
    endif()
    
    # Install MPI library
    install(TARGETS MPIBagging_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    
    message(STATUS "MPI Bagging support enabled")
endif()
# 决策树核心模块，包含 criterion、finder、pruner、trainer、ensemble
add_library(DecisionTree_lib
    # 分裂准则
    criterion/MSECriterion.cpp
    criterion/MAECriterion.cpp
    criterion/HuberCriterion.cpp
    criterion/QuantileCriterion.cpp
    criterion/LogCoshCriterion.cpp
    criterion/PoissonCriterion.cpp
    
    # **优化的分割器** - 使用预计算直方图
    finder/ExhaustiveSplitFinder.cpp
    finder/RandomSplitFinder.cpp
    finder/QuartileSplitFinder.cpp
    finder/HistogramEWFinder.cpp        # 等宽直方图
    finder/HistogramEQFinder.cpp        # 等频直方图
    finder/AdaptiveEWFinder.cpp         # 自适应等宽
    finder/AdaptiveEQFinder.cpp         # 自适应等频
    
    # 剪枝器
    pruner/NoPruner.cpp
    pruner/MinGainPrePruner.cpp
    pruner/CostComplexityPruner.cpp
    pruner/ReducedErrorPruner.cpp
    
    # 训练器
    trainer/SingleTreeTrainer.cpp
    
    # 集成方法
    ensemble/BaggingTrainer.cpp
)

target_include_directories(DecisionTree_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# **重要**: 链接预计算直方图优化库
target_link_libraries(DecisionTree_lib PUBLIC
    HistogramOptimized_lib              # 预计算直方图优化
)

# **OpenMP配置** - 避免重复定义
if(OpenMP_CXX_FOUND)
    target_link_libraries(DecisionTree_lib PUBLIC OpenMP::OpenMP_CXX)
    # 不手动定义 _OPENMP，让OpenMP自动处理
endif()

# 性能优化编译选项（静默版本）
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(DecisionTree_lib PRIVATE
        -O3                    # 最高优化级别
        -march=native          # CPU特定优化
        -funroll-loops         # 循环展开
        -ftree-vectorize      # 自动向量化
        -ffast-math           # 快速数学运算
        # 移除verbose优化报告
    )
endif()

# MSVC优化
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(DecisionTree_lib PRIVATE
        /O2                    # 最大优化
        /arch:AVX2            # AVX2指令集
        /fp:fast              # 快速浮点运算
    )
endif()

# **警告控制** - 减少编译输出噪音
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(DecisionTree_lib PRIVATE
        -Wall                  # 基本警告
        -Wno-unused-parameter  # 忽略未使用参数警告（接口兼容性）
        -Wno-unused-variable   # 忽略未使用变量警告
        -Wno-sign-compare      # 忽略符号比较警告
        -Wno-maybe-uninitialized  # 忽略可能未初始化警告
    )
endif()

# 条件编译选项
option(ENABLE_TREE_PROFILING "Enable decision tree performance profiling" OFF)
if(ENABLE_TREE_PROFILING)
    target_compile_definitions(DecisionTree_lib PUBLIC TREE_PROFILING)
endif()

# **直方图优化级别控制**
option(HISTOGRAM_OPTIMIZATION_LEVEL "Histogram optimization level (0-3)" 2)
target_compile_definitions(DecisionTree_lib PUBLIC 
    HISTOGRAM_OPT_LEVEL=${HISTOGRAM_OPTIMIZATION_LEVEL}
)

# 内存优化选项
option(ENABLE_TREE_MEMORY_POOL "Enable memory pool for tree nodes" ON)
if(ENABLE_TREE_MEMORY_POOL)
    target_compile_definitions(DecisionTree_lib PUBLIC TREE_MEMORY_POOL)
endif()

# Debug模式特殊处理
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(DecisionTree_lib PUBLIC TREE_DEBUG)
    # Debug模式下使用较低优化等级
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(DecisionTree_lib PRIVATE -g -O1)
    endif()
else()
    # Release模式性能监控
    target_compile_definitions(DecisionTree_lib PUBLIC TREE_PERFORMANCE_MODE)
endif()

# **链接时优化**（Release模式）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(DecisionTree_lib PRIVATE -flto)
        target_link_options(DecisionTree_lib PRIVATE -flto)
    endif()
endif()

# 确保C++17标准
target_compile_features(DecisionTree_lib PUBLIC cxx_std_17)