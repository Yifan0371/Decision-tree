find_package(MPI REQUIRED)

add_executable(MPIBaggingMain
    main.cpp
    ${PROJECT_SOURCE_DIR}/src/tree/ensemble/MPIBaggingTrainer.cpp
)

target_include_directories(MPIBaggingMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(MPIBaggingMain PRIVATE
    DecisionTree_lib    # 新增：链接包含 BaggingTrainer 实现的核心库
    DataIO_lib
    DataSplit_lib
    MPI::MPI_CXX
)

target_compile_options(MPIBaggingMain PRIVATE
    ${MPI_CXX_COMPILE_FLAGS}
)
set_target_properties(MPIBaggingMain PROPERTIES
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
)

install(TARGETS MPIBaggingMain RUNTIME DESTINATION bin)

add_custom_target(mpi-bagging-info
    COMMAND ${CMAKE_COMMAND} -E echo "=== MPI Bagging Usage ==="
    COMMAND ${CMAKE_COMMAND} -E echo "Run: mpirun -np <num_procs> ./MPIBaggingMain <args>"
)
