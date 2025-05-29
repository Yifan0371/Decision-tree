# DataIO 模块
add_library(DataIO_lib
    DataIO.cpp
)

target_include_directories(DataIO_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
