#!/bin/bash
# script/boosting/gbrt/test_dart_weight_strategies.sh

echo "=== DART权重策略对比测试 ==="

DATA_PATH="data/data_clean/cleaned_data.csv"
EXECUTABLE="build/RegressionBoostingMain"

echo "测试不同权重策略的效果："

echo "1. 不做权重调整 (对照组):"
$EXECUTABLE "$DATA_PATH" "squared" 40 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.2 "false" "false"

echo -e "\n2. 温和权重调整:"
# 需要修改代码支持权重策略参数
$EXECUTABLE "$DATA_PATH" "squared" 40 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.2 "true" "false"

echo -e "\n3. 最优配置 (深度6 + 高学习率 + 无权重调整):"
$EXECUTABLE "$DATA_PATH" "squared" 40 0.15 6 1 "mse" "exhaustive" 0.8 "true" 0.2 "false" "false"

echo -e "\n4. 最优配置变体 (更多迭代):"
$EXECUTABLE "$DATA_PATH" "squared" 60 0.1 6 1 "mse" "exhaustive" 0.9 "true" 0.15 "false" "false"