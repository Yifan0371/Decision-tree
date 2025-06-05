# src/app/CMakeLists.txt

# -------- 1. SingleTreeApp_lib （单棵树应用库） --------
add_library(SingleTreeApp_lib
    SingleTreeApp.cpp
)
target_include_directories(SingleTreeApp_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(SingleTreeApp_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# -------- 2. BaggingApp_lib （Bagging 应用库） --------
add_library(BaggingApp_lib
    BaggingApp.cpp
)
target_include_directories(BaggingApp_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(BaggingApp_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    OpenMP::OpenMP_CXX
)

# -------- 3. MPIBaggingApp_lib （MPI+OpenMP Bagging 库） --------
find_package(MPI REQUIRED)
if(MPI_FOUND)
    add_library(MPIBaggingApp_lib
        MPIBaggingApp.cpp
    )
    target_include_directories(MPIBaggingApp_lib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )
    target_link_libraries(MPIBaggingApp_lib PUBLIC
        DecisionTree_lib
        DataIO_lib
        DataSplit_lib
        OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES}
    )
    # 避免 MPI C++ 旧接口警告
    target_compile_definitions(MPIBaggingApp_lib PUBLIC OMPI_SKIP_MPICXX=1)
else()
    message(WARNING "MPI not found - MPIBaggingApp_lib will NOT be built")
endif()
