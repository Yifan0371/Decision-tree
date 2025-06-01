# LightGBM核心库
add_library(LightGBM_lib
    # 采样器
    sampling/GOSSSampler.cpp
    
    # 特征绑定
    feature/FeatureBundler.cpp
    
    # 树构建器
    tree/LeafwiseTreeBuilder.cpp
    
    # 训练器
    trainer/LightGBMTrainer.cpp
    
    # 应用
    app/LightGBMApp.cpp
)

target_include_directories(LightGBM_lib PUBLIC
    \${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(LightGBM_lib PUBLIC
    DecisionTree_lib
    RegressionBoosting_lib  # 复用损失函数
    DataIO_lib
    DataSplit_lib
)
