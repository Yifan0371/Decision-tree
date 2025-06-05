# src/CMakeLists.txt

# 将所有可执行文件输出到 build/ 根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 各模块子目录
add_subdirectory(preprocessing)
add_subdirectory(functions/io)
add_subdirectory(pipeline)
add_subdirectory(tree)
add_subdirectory(boosting)
add_subdirectory(xgboost)
add_subdirectory(lightgbm)
add_subdirectory(app)
add_subdirectory(histogram)

# 如果有性能测试（BUILD_PERFORMANCE_TESTS = ON），将它们纳入
if(BUILD_PERFORMANCE_TESTS)
    add_subdirectory(tests/performance)
endif()
