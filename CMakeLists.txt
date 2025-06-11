# CMakeLists.txt - 增强OpenMP支持和并行优化
cmake_minimum_required(VERSION 3.10)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add this section after OpenMP configuration in the root CMakeLists.txt
if(ENABLE_MPI AND MPI_CXX_FOUND)
    add_subdirectory(mpi_bagging)
endif() 
# =============================================================================
# MPI Configuration (Optional)
# =============================================================================
option(ENABLE_MPI "Enable MPI support for distributed Bagging" OFF)

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found:")
        message(STATUS "  MPI Version: ${MPI_CXX_VERSION}")
        message(STATUS "  MPI Compiler: ${MPI_CXX_COMPILER}")
        message(STATUS "  MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI Libraries: ${MPI_CXX_LIBRARIES}")
        
        # Check MPI+OpenMP compatibility
        if(OpenMP_CXX_FOUND)
            message(STATUS "MPI+OpenMP hybrid parallelism enabled")
        endif()
    endif()
endif()

# Add to configuration summary at the end:
message(STATUS "MPI Support: ${ENABLE_MPI}")
if(ENABLE_MPI AND MPI_CXX_FOUND)
    message(STATUS "MPI Implementation: ${MPI_CXX_VERSION}")
endif()
# =============================================================================
# OpenMP配置（增强版）
# =============================================================================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenMP not found! Please install OpenMP support.")
endif()

# =============================================================================
# 编译器优化配置
# =============================================================================

# 基础优化选项
set(BASE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")

# 针对不同编译器的OpenMP优化
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC特定优化
    set(COMPILER_SPECIFIC_FLAGS "-fopenmp-simd -ffast-math -funroll-loops")
    message(STATUS "Using GCC-specific optimizations")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang特定优化
    set(COMPILER_SPECIFIC_FLAGS "-fopenmp-simd -ffast-math -funroll-loops")
    message(STATUS "Using Clang-specific optimizations")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC特定优化
    set(COMPILER_SPECIFIC_FLAGS "/fp:fast /arch:AVX2")
    message(STATUS "Using MSVC-specific optimizations")
endif()

# 组合所有编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_CXX_FLAGS} ${COMPILER_SPECIFIC_FLAGS}")

# =============================================================================
# OpenMP链接配置
# =============================================================================

# 确保所有目标都链接OpenMP
link_libraries(OpenMP::OpenMP_CXX)

# 设置OpenMP运行时环境变量（可选）
set(ENV{OMP_NUM_THREADS} "auto")
set(ENV{OMP_DYNAMIC} "true")
set(ENV{OMP_NESTED} "false")

# =============================================================================
# 性能分析支持（可选）
# =============================================================================

option(ENABLE_PROFILING "Enable profiling support" OFF)
if(ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
    message(STATUS "Profiling enabled")
endif()

option(ENABLE_TIMING "Enable detailed timing measurements" OFF)
if(ENABLE_TIMING)
    add_definitions(-DENABLE_TIMING)
    message(STATUS "Timing measurements enabled")
endif()

# =============================================================================
# 并行性能测试支持
# =============================================================================

option(BUILD_PERFORMANCE_TESTS "Build performance test executables" OFF)
if(BUILD_PERFORMANCE_TESTS)
    message(STATUS "Performance tests will be built")
    add_definitions(-DBUILD_PERFORMANCE_TESTS)
endif()

# =============================================================================
# 头文件和子目录
# =============================================================================

# 把 include/ 加到头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 将 src 子目录也纳入构建
add_subdirectory(src)

# =============================================================================
# 性能测试可执行文件（如果启用）
# =============================================================================

if(BUILD_PERFORMANCE_TESTS)
    # 并行性能测试
    add_executable(ParallelPerformanceTest
        tests/performance/parallel_performance_test.cpp
    )
    target_link_libraries(ParallelPerformanceTest PRIVATE
        DecisionTree_lib
        RegressionBoosting_lib
        XGBoost_lib
        LightGBM_lib
        DataIO_lib
        DataSplit_lib
    )
    
    # Boosting性能对比测试
    add_executable(BoostingBenchmark
        tests/performance/boosting_benchmark.cpp
    )
    target_link_libraries(BoostingBenchmark PRIVATE
        DecisionTree_lib
        RegressionBoosting_lib
        XGBoost_lib
        LightGBM_lib
        DataIO_lib
        DataSplit_lib
    )
endif()

# =============================================================================
# 安装配置
# =============================================================================

# 安装可执行文件
#install(TARGETS DecisionTreeMain RegressionBoostingMain XGBoostMain LightGBMMain
#        RUNTIME DESTINATION bin)

# 安装库文件
install(TARGETS DecisionTree_lib RegressionBoosting_lib XGBoost_lib LightGBM_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# 安装头文件
install(DIRECTORY include/ DESTINATION include)

# =============================================================================
# 配置摘要
# =============================================================================

message(STATUS "=== Build Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenMP Support: ${OpenMP_CXX_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP Version: ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "Profiling: ${ENABLE_PROFILING}")
message(STATUS "Timing: ${ENABLE_TIMING}")
message(STATUS "Performance Tests: ${BUILD_PERFORMANCE_TESTS}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=====================================")

# =============================================================================
# 编译提示
# =============================================================================

# 添加自定义目标来显示并行编译提示
add_custom_target(parallel-info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Parallel Compilation Tips ==="
    COMMAND ${CMAKE_COMMAND} -E echo "For faster compilation, use: make -j\$(nproc)"
    COMMAND ${CMAKE_COMMAND} -E echo "For ninja: ninja -j\$(nproc)"
    COMMAND ${CMAKE_COMMAND} -E echo "OpenMP will automatically use available CPU cores at runtime"
    COMMAND ${CMAKE_COMMAND} -E echo "Set OMP_NUM_THREADS environment variable to control thread count"
    COMMAND ${CMAKE_COMMAND} -E echo "================================="
)
# =================================================================
# Main可执行文件
# =================================================================

# 添加main子目录
add_subdirectory(main)
