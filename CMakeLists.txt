cmake_minimum_required(VERSION 3.10)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# 1. OpenMP配置（增强版）
# =============================================================================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")

    # 基础优化选项
    set(BASE_CXX_FLAGS "-O3 -march=native -Wall -Wextra")
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

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_CXX_FLAGS} ${COMPILER_SPECIFIC_FLAGS}")
    # 确保所有目标都链接OpenMP
    link_libraries(OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found! Please install OpenMP support.")
endif()

# 设置OpenMP运行时环境变量（可选）
set(ENV{OMP_NUM_THREADS} "auto")
set(ENV{OMP_DYNAMIC} "true")
set(ENV{OMP_NESTED} "false")

# =============================================================================
# 2. 性能分析支持（可选）
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
# 3. 并行性能测试支持
# =============================================================================
option(BUILD_PERFORMANCE_TESTS "Build performance test executables" OFF)
if(BUILD_PERFORMANCE_TESTS)
    message(STATUS "Performance tests will be built")
    add_definitions(-DBUILD_PERFORMANCE_TESTS)
endif()

# =============================================================================
# 4. MPI配置
# =============================================================================
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI Found: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "MPI not found! Please install MPI (OpenMPI/MPICH).")
endif()

# =============================================================================
# 5. 头文件和子目录
# =============================================================================
# 把 include/ 加到头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 将 src 子目录纳入构建（生成各个库）
add_subdirectory(src)

# =============================================================================
# 6. 性能测试可执行文件（如果启用）
# =============================================================================
if(BUILD_PERFORMANCE_TESTS)
    # 并行性能测试
    add_executable(ParallelPerformanceTest
        ${PROJECT_SOURCE_DIR}/tests/performance/parallel_performance_test.cpp
    )
    target_link_libraries(ParallelPerformanceTest PRIVATE
        DecisionTree_lib
        RegressionBoosting_lib
        XGBoost_lib
        LightGBM_lib
        DataIO_lib
        DataSplit_lib
        OpenMP::OpenMP_CXX
    )
    
    # Boosting性能对比测试
    add_executable(BoostingBenchmark
        ${PROJECT_SOURCE_DIR}/tests/performance/boosting_benchmark.cpp
    )
    target_link_libraries(BoostingBenchmark PRIVATE
        DecisionTree_lib
        RegressionBoosting_lib
        XGBoost_lib
        LightGBM_lib
        DataIO_lib
        DataSplit_lib
        OpenMP::OpenMP_CXX
    )
endif()

# =============================================================================
# 7. “其他可执行” 构建
#    在顶层直接声明，不再add_subdirectory(main)
# =============================================================================

# —— 7.1 SingleTreeMain: 单棵树示例 —— 
add_executable(SingleTreeMain
    ${PROJECT_SOURCE_DIR}/main/single_tree/main.cpp
)
target_include_directories(SingleTreeMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(SingleTreeMain PRIVATE
    SingleTreeApp_lib
    BaggingApp_lib         # 为了 runBaggingApp
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# —— 7.2 BaggingMain: 单机 Bagging 示例 —— 
add_executable(BaggingMain
    ${PROJECT_SOURCE_DIR}/main/bagging/main.cpp
)
target_include_directories(BaggingMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(BaggingMain PRIVATE
    BaggingApp_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# —— 7.3 RegressionBoostingMain: 回归 Boosting 示例 —— 
add_executable(RegressionBoostingMain
    ${PROJECT_SOURCE_DIR}/main/gbrt/main.cpp
)
target_include_directories(RegressionBoostingMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(RegressionBoostingMain PRIVATE
    RegressionBoosting_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# —— 7.4 XGBoostMain: XGBoost 示例 —— 
add_executable(XGBoostMain
    ${PROJECT_SOURCE_DIR}/main/xgboost/main.cpp
)
target_include_directories(XGBoostMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(XGBoostMain PRIVATE
    XGBoost_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# —— 7.5 LightGBMMain: LightGBM 示例 —— 
add_executable(LightGBMMain
    ${PROJECT_SOURCE_DIR}/main/lightgbm/main.cpp
)
target_include_directories(LightGBMMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(LightGBMMain PRIVATE
    LightGBM_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# —— 7.6 MPIBaggingMain: MPI+OpenMP Bagging 示例 —— 
add_executable(MPIBaggingMain
    ${PROJECT_SOURCE_DIR}/main/mpi_bagging/main.cpp
)
target_include_directories(MPIBaggingMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)
target_link_libraries(MPIBaggingMain PRIVATE
    MPIBaggingApp_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
    ${MPI_CXX_LIBRARIES}
)
# 避免 MPI C++ 旧接口警告
target_compile_definitions(MPIBaggingMain PRIVATE OMPI_SKIP_MPICXX=1)

# （可选）创建一个软链接 mpi_bagging 方便脚本调用
add_custom_target(mpi_bagging_link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/MPIBaggingMain
        ${CMAKE_BINARY_DIR}/mpi_bagging
    DEPENDS MPIBaggingMain
    COMMENT "Creating mpi_bagging symlink"
)

# =============================================================================
# 8. 安装配置
# =============================================================================

# 安装库文件
install(TARGETS DecisionTree_lib RegressionBoosting_lib XGBoost_lib LightGBM_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(TARGETS DataIO_lib DataSplit_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# 安装头文件
install(DIRECTORY include/ DESTINATION include)

# =============================================================================
# 9. 配置摘要 & 编译提示
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
message(STATUS "MPI Support: ${MPI_FOUND} (${MPI_CXX_COMPILER})")
message(STATUS "Profiling: ${ENABLE_PROFILING}")
message(STATUS "Timing: ${ENABLE_TIMING}")
message(STATUS "Performance Tests: ${BUILD_PERFORMANCE_TESTS}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=====================================")

add_custom_target(parallel-info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Parallel Compilation Tips ==="
    COMMAND ${CMAKE_COMMAND} -E echo "For faster compilation, use: make -j\$(nproc)"
    COMMAND ${CMAKE_COMMAND} -E echo "For ninja: ninja -j\$(nproc)"
    COMMAND ${CMAKE_COMMAND} -E echo "OpenMP will automatically use available CPU cores at runtime"
    COMMAND ${CMAKE_COMMAND} -E echo "Set OMP_NUM_THREADS environment variable to control thread count"
    COMMAND ${CMAKE_COMMAND} -E echo "================================="
)
