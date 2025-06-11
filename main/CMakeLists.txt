# Put all executables in top-level build dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# -----------------------------------------------------------------------------
# MPI Bagging (always included, but requires MPI)
# -----------------------------------------------------------------------------
add_subdirectory(mpi_bagging)

# -----------------------------------------------------------------------------
# Other main executables
# -----------------------------------------------------------------------------
add_executable(DecisionTreeMain single_tree/main.cpp)
target_link_libraries(DecisionTreeMain PRIVATE
    DataIO_lib DataSplit_lib DecisionTree_lib SingleTreeApp_lib BaggingApp_lib
)

add_executable(BaggingMain bagging/main.cpp)
target_link_libraries(BaggingMain PRIVATE
    BaggingApp_lib DecisionTree_lib DataIO_lib DataSplit_lib
)

add_executable(RegressionBoostingMain gbrt/main.cpp)
target_link_libraries(RegressionBoostingMain PRIVATE
    DataIO_lib DataSplit_lib DecisionTree_lib RegressionBoosting_lib
)

add_executable(XGBoostMain xgboost/main.cpp)
target_link_libraries(XGBoostMain PRIVATE
    DataIO_lib DataSplit_lib DecisionTree_lib RegressionBoosting_lib XGBoost_lib
)

add_executable(LightGBMMain lightgbm/main.cpp)
target_link_libraries(LightGBMMain PRIVATE
    DataIO_lib DataSplit_lib DecisionTree_lib RegressionBoosting_lib LightGBM_lib
)

add_executable(DataCleanApp data_clean/main.cpp)
target_link_libraries(DataCleanApp PRIVATE
    DataCleaner_lib
)

# -----------------------------------------------------------------------------
# Usage info
# -----------------------------------------------------------------------------
add_custom_target(main-info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Available Main Programs ==="
    COMMAND ${CMAKE_COMMAND} -E echo "DecisionTreeMain"
    COMMAND ${CMAKE_COMMAND} -E echo "BaggingMain"
    COMMAND ${CMAKE_COMMAND} -E echo "RegressionBoostingMain"
    COMMAND ${CMAKE_COMMAND} -E echo "XGBoostMain"
    COMMAND ${CMAKE_COMMAND} -E echo "LightGBMMain"
    COMMAND ${CMAKE_COMMAND} -E echo "DataCleanApp"
    COMMAND ${CMAKE_COMMAND} -E echo "MPIBaggingMain (in mpi_bagging/)"
    COMMAND ${CMAKE_COMMAND} -E echo "================================"
)

# -----------------------------------------------------------------------------
# Install (optional)
# -----------------------------------------------------------------------------
install(TARGETS
    DecisionTreeMain BaggingMain RegressionBoostingMain
    XGBoostMain LightGBMMain DataCleanApp MPIBaggingMain
    RUNTIME DESTINATION bin
)
