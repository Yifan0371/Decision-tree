# src/preprocessing/CMakeLists.txt
add_library(DataCleaner_lib
    DataCleaner.cpp
)

target_include_directories(DataCleaner_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
