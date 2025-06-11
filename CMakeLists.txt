cmake_minimum_required(VERSION 3.10)
project(DecisionTree VERSION 1.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# 默认开启 MPI
option(ENABLE_MPI "Enable MPI support for distributed Bagging" ON)

# -----------------------------------------------------------------------------
# Find OpenMP (always required)
# -----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found! Please install OpenMP support.")
endif()
message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")

# -----------------------------------------------------------------------------
# Find MPI (only if ENABLE_MPI)
# -----------------------------------------------------------------------------
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_VERSION}")
    endif()
endif()

# -----------------------------------------------------------------------------
# Include common headers
# -----------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(main)
