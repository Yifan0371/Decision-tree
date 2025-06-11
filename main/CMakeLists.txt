# main/CMakeLists.txt - 所有主程序可执行文件

# 将所有可执行文件输出到 build/ 根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Add this section at the end of main/CMakeLists.txt

# =================================================================
# MPI Bagging (Optional)
# =================================================================
if(ENABLE_MPI)
    add_subdirectory(mpi_bagging)
endif()

# Update the usage information section:
add_custom_target(main-info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Available Main Programs ==="
    COMMAND ${CMAKE_COMMAND} -E echo "DecisionTreeMain    - Single tree + Bagging modes"
    COMMAND ${CMAKE_COMMAND} -E echo "BaggingMain         - Bootstrap Aggregating only"
    COMMAND ${CMAKE_COMMAND} -E echo "RegressionBoostingMain - GBRT with DART support"
    COMMAND ${CMAKE_COMMAND} -E echo "XGBoostMain         - XGBoost implementation"
    COMMAND ${CMAKE_COMMAND} -E echo "LightGBMMain        - LightGBM implementation"
    COMMAND ${CMAKE_COMMAND} -E echo "DataCleanApp        - Data preprocessing tool"
    COMMAND ${CMAKE_COMMAND} -E echo "$<$<BOOL:${ENABLE_MPI}>:MPIBaggingMain       - MPI distributed Bagging>"
    COMMAND ${CMAKE_COMMAND} -E echo "================================"
    COMMAND ${CMAKE_COMMAND} -E echo "Usage: ./build/[ProgramName] --help"
    COMMAND ${CMAKE_COMMAND} -E echo "$<$<BOOL:${ENABLE_MPI}>:MPI Usage: mpirun -np N ./build/MPIBaggingMain [args]>"
)
# =================================================================
# 单决策树主程序 (包含Bagging模式)
# =================================================================
add_executable(DecisionTreeMain
    single_tree/main.cpp
)
target_link_libraries(DecisionTreeMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    SingleTreeApp_lib
    BaggingApp_lib
)

# =================================================================
# 独立Bagging主程序
# =================================================================
add_executable(BaggingMain
    bagging/main.cpp
)
target_link_libraries(BaggingMain PRIVATE
    BaggingApp_lib
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)

# =================================================================
# GBRT主程序
# =================================================================
add_executable(RegressionBoostingMain
    gbrt/main.cpp
)
target_link_libraries(RegressionBoostingMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
)

# =================================================================
# XGBoost主程序
# =================================================================
add_executable(XGBoostMain
    xgboost/main.cpp
)
target_link_libraries(XGBoostMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
    XGBoost_lib
)

# =================================================================
# LightGBM主程序
# =================================================================
add_executable(LightGBMMain
    lightgbm/main.cpp
)
target_link_libraries(LightGBMMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    RegressionBoosting_lib
    LightGBM_lib
)

# =================================================================
# 数据清洗工具
# =================================================================
add_executable(DataCleanApp
    data_clean/main.cpp
)
target_link_libraries(DataCleanApp PRIVATE
    DataCleaner_lib
)

# =================================================================
# 安装配置
# =================================================================
install(TARGETS 
    DecisionTreeMain 
    BaggingMain
    RegressionBoostingMain 
    XGBoostMain 
    LightGBMMain
    DataCleanApp
    RUNTIME DESTINATION bin
)
