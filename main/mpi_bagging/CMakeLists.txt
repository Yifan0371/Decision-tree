# main/mpi_bagging/CMakeLists.txt

# Only build if MPI is enabled
if(ENABLE_MPI AND MPI_CXX_FOUND)
    # MPI Bagging executable
    add_executable(MPIBaggingMain
        main.cpp
    )
    
    # Include directories
    target_include_directories(MPIBaggingMain PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )
    
    # Link libraries
    target_link_libraries(MPIBaggingMain PRIVATE
        MPIBagging_lib
        DataIO_lib
        DataSplit_lib
        MPI::MPI_CXX
    )
    
    # MPI compile flags
    target_compile_options(MPIBaggingMain PRIVATE
        ${MPI_CXX_COMPILE_FLAGS}
    )
    
    # MPI link flags
    set_target_properties(MPIBaggingMain PROPERTIES
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
    )
    
    # Install
    install(TARGETS MPIBaggingMain
        RUNTIME DESTINATION bin
    )
    
    # Usage information
    add_custom_target(mpi-bagging-info
        COMMAND ${CMAKE_COMMAND} -E echo "=== MPI Bagging Usage ==="
        COMMAND ${CMAKE_COMMAND} -E echo "Build: cmake -DENABLE_MPI=ON .."
        COMMAND ${CMAKE_COMMAND} -E echo "Run: mpirun -np <num_processes> ./MPIBaggingMain <args>"
        COMMAND ${CMAKE_COMMAND} -E echo "Example: mpirun -np 4 ./MPIBaggingMain data.csv 100"
        COMMAND ${CMAKE_COMMAND} -E echo "========================="
    )
endif()