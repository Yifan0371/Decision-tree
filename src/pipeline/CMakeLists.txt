# DataSplit 模块
add_library(DataSplit_lib
    DataSplit.cpp
)

target_include_directories(DataSplit_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# DataSplit 依赖 DataIO
target_link_libraries(DataSplit_lib PUBLIC
    DataIO_lib
)
