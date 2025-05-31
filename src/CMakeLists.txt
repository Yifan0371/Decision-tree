# src/CMakeLists.txt

# 把可执行文件直接输出到顶层 build 目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# 各模块子目录
add_subdirectory(preprocessing)
add_subdirectory(functions/io)
add_subdirectory(pipeline)
add_subdirectory(tree)
add_subdirectory(app)

# 原有的单树可执行文件
add_executable(DecisionTreeMain
    main.cpp
)

# 链接各模块库
target_link_libraries(DecisionTreeMain PRIVATE
    DataIO_lib
    DataSplit_lib
    DecisionTree_lib
    SingleTreeApp_lib
    BaggingApp_lib
)