#!/bin/bash

# =============================================================================
# script/boosting/gbrt/test_gbrt_comprehensive.sh
# GBRT (Gradient Boosted Regression Trees) 全面参数测试脚本
# 在项目根目录运行: bash script/boosting/gbrt/test_gbrt_comprehensive.sh
# =============================================================================

DATA_PATH="data/data_clean/cleaned_data.csv"
EXECUTABLE="build/RegressionBoostingMain"

echo "=== GBRT 全面参数测试 ==="
echo "数据文件: $DATA_PATH"
echo "可执行文件: $EXECUTABLE"
echo ""

# 检查文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 找不到可执行文件 $EXECUTABLE"
    echo "请先运行: cd build && make"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 找不到数据文件 $DATA_PATH"
    echo "请检查数据文件是否存在，或使用其他数据文件"
    # 尝试其他可能的数据文件
    for alt_data in "data/data_clean/cleaned_sample_100_rows.csv" "data/data_clean/cleaned_sample_400_rows.csv" "data/data_clean/cleaned_15k_random.csv"; do
        if [ -f "$alt_data" ]; then
            echo "使用替代数据文件: $alt_data"
            DATA_PATH="$alt_data"
            break
        fi
    done
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "未找到任何可用的数据文件，退出"
        exit 1
    fi
fi

echo "使用数据文件: $DATA_PATH"
echo ""

# 解析输出结果的函数
extract_metrics() {
    local output="$1"
    local test_mse=$(echo "$output" | grep "Test MSE:" | grep -o "[0-9.-]*" | tail -1)
    local train_mse=$(echo "$output" | grep "Train MSE:" | grep -o "[0-9.-]*" | tail -1)
    local train_time=$(echo "$output" | grep -o "Train Time: [0-9]*ms" | grep -o "[0-9]*" | head -1)
    
    # 处理空值和NaN
    [ -z "$test_mse" ] || [ "$test_mse" = "-nan" ] && test_mse="ERROR"
    [ -z "$train_mse" ] || [ "$train_mse" = "-nan" ] && train_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    
    echo "$test_mse,$train_mse,$train_time"
}

# 运行单个测试
run_test() {
    local desc="$1"
    local loss="$2"
    local iterations="$3"
    local lr="$4"
    local depth="$5"
    local min_leaf="$6"
    local criterion="$7"
    local split_method="$8"
    local subsample="$9"
    
    local start_time=$(date +%s%3N)
    local output
    
    # 执行命令并捕获输出
    output=$($EXECUTABLE "$DATA_PATH" "$loss" $iterations $lr $depth $min_leaf "$criterion" "$split_method" $subsample 2>&1)
    local exit_code=$?
    
    local end_time=$(date +%s%3N)
    local wall_time=$((end_time - start_time))
    
    # 如果程序执行失败，输出错误信息
    if [ $exit_code -ne 0 ]; then
        printf "%-25s | %-8s | %-8s | %-8s | %s [ERROR: exit code %d]\n" "$desc" "ERROR" "0ms" "${wall_time}ms" "$loss $iterations $lr $depth $min_leaf" "$exit_code"
        return
    fi
    
    local metrics=$(extract_metrics "$output")
    local test_mse=$(echo "$metrics" | cut -d',' -f1)
    local train_mse=$(echo "$metrics" | cut -d',' -f2)
    local train_time=$(echo "$metrics" | cut -d',' -f3)
    
    printf "%-25s | %-8s | %-8s | %-8s | %s\n" "$desc" "$test_mse" "${train_time}ms" "${wall_time}ms" "$loss $iterations $lr $depth $min_leaf $criterion $split_method $subsample"
}

# 打印表头
printf "%-25s | %-8s | %-8s | %-8s | %s\n" "测试描述" "TestMSE" "TrainTime" "WallTime" "参数配置"
echo "--------------------------------------------------------------------------------------------------------"

echo "=== 1. 损失函数测试 ==="
run_test "Squared Loss (基准)" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "Huber Loss (鲁棒)" "huber" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "Absolute Loss (L1)" "absolute" 30 0.1 4 1 "mae" "exhaustive" 1.0
run_test "Quantile Loss (分位数)" "quantile" 30 0.1 4 1 "mae" "exhaustive" 1.0

echo ""
echo "=== 2. 迭代次数测试 ==="
run_test "迭代10次" "squared" 10 0.1 4 1 "mse" "exhaustive" 1.0
run_test "迭代25次" "squared" 25 0.1 4 1 "mse" "exhaustive" 1.0
run_test "迭代50次" "squared" 50 0.1 4 1 "mse" "exhaustive" 1.0
run_test "迭代100次" "squared" 100 0.1 4 1 "mse" "exhaustive" 1.0

echo ""
echo "=== 3. 学习率测试 ==="
run_test "学习率0.01" "squared" 30 0.01 4 1 "mse" "exhaustive" 1.0
run_test "学习率0.05" "squared" 30 0.05 4 1 "mse" "exhaustive" 1.0
run_test "学习率0.1" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "学习率0.2" "squared" 30 0.2 4 1 "mse" "exhaustive" 1.0
run_test "学习率0.5" "squared" 30 0.5 4 1 "mse" "exhaustive" 1.0

echo ""
echo "=== 4. 最大深度测试 ==="
run_test "深度2" "squared" 30 0.1 2 1 "mse" "exhaustive" 1.0
run_test "深度3" "squared" 30 0.1 3 1 "mse" "exhaustive" 1.0
run_test "深度4" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "深度6" "squared" 30 0.1 6 1 "mse" "exhaustive" 1.0

echo ""
echo "=== 5. 最小叶子样本测试 ==="
run_test "最小叶子1" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "最小叶子2" "squared" 30 0.1 4 2 "mse" "exhaustive" 1.0
run_test "最小叶子5" "squared" 30 0.1 4 5 "mse" "exhaustive" 1.0
run_test "最小叶子10" "squared" 30 0.1 4 10 "mse" "exhaustive" 1.0

echo ""
echo "=== 6. 分割准则测试 ==="
run_test "MSE准则" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "MAE准则" "squared" 30 0.1 4 1 "mae" "exhaustive" 1.0
run_test "Huber准则" "squared" 30 0.1 4 1 "huber" "exhaustive" 1.0

echo ""
echo "=== 7. 分割方法测试 ==="
run_test "穷举分割" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_test "随机分割" "squared" 30 0.1 4 1 "mse" "random" 1.0
run_test "等宽直方图" "squared" 30 0.1 4 1 "mse" "histogram_ew" 1.0
run_test "等频直方图" "squared" 30 0.1 4 1 "mse" "histogram_eq" 1.0

echo ""
echo "=== 8. 子采样率测试 ==="
run_test "子采样50%" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.5
run_test "子采样70%" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.7
run_test "子采样80%" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.8
run_test "无子采样" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0

echo ""
echo "=== 9. 组合优化测试 ==="
run_test "最优组合1" "squared" 50 0.1 6 1 "mse" "exhaustive" 0.8
run_test "最优组合2" "huber" 50 0.05 4 2 "mse" "exhaustive" 0.9
run_test "快速组合" "squared" 20 0.2 3 1 "mse" "random" 1.0
run_test "鲁棒组合" "absolute" 40 0.1 4 3 "mae" "exhaustive" 0.8
run_test "平衡组合" "squared" 30 0.15 4 2 "mse" "histogram_ew" 0.85

echo ""
echo "=== GBRT 全面测试完成 ==="

# 简单的调试测试
echo ""
echo "=== 调试信息 ==="
echo "直接运行一个简单测试:"
$EXECUTABLE "$DATA_PATH" "squared" 10 0.1 3 1 "mse" "exhaustive" 1.0