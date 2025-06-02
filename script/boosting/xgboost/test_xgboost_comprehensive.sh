#!/bin/bash

# =============================================================================
# script/boosting/xgboost/test_xgboost_comprehensive.sh
# XGBoost 全面参数测试脚本
# 在项目根目录运行: bash script/boosting/xgboost/test_xgboost_comprehensive.sh
# =============================================================================

DATA_PATH="/data/data_base/cleaned_data.csv"
EXECUTABLE="build/XGBoostMain"

# 结果文件路径
RESULTS_DIR="script/boosting/xgboost"
RESULTS_FILE="$RESULTS_DIR/xgboost_comprehensive_results.txt"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 清空之前的结果文件
> "$RESULTS_FILE"

# 记录开始时间
START_TIME=$(date)

# 输出函数：同时输出到控制台和文件
output() {
    echo "$1" | tee -a "$RESULTS_FILE"
}

# 简洁输出到文件
output_to_file_only() {
    echo "$1" >> "$RESULTS_FILE"
}

# 控制台输出
output_to_console_only() {
    echo "$1"
}

output "=== XGBoost 全面参数测试 ==="
output "开始时间: $START_TIME"
output "数据文件: $DATA_PATH"
output "可执行文件: $EXECUTABLE"
output "结果保存到: $RESULTS_FILE"
output ""

# 检查文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    output "错误: 找不到可执行文件 $EXECUTABLE"
    output "请先运行: cd build && make"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    output "错误: 找不到数据文件 $DATA_PATH"
    output "请检查数据文件是否存在，或使用其他数据文件"
    # 尝试其他可能的数据文件
    for alt_data in "data/data_clean/cleaned_data.csv" "data/data_clean/cleaned_sample_400_rows.csv" "data/data_clean/cleaned_15k_random.csv"; do
        if [ -f "$alt_data" ]; then
            output "使用替代数据文件: $alt_data"
            DATA_PATH="$alt_data"
            break
        fi
    done
    
    if [ ! -f "$DATA_PATH" ]; then
        output "未找到任何可用的数据文件，退出"
        exit 1
    fi
fi

output "使用数据文件: $DATA_PATH"
output ""

# 解析输出结果的函数
extract_metrics() {
    local output="$1"
    # 更精确的MSE提取，使用sed
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_mse=$(echo "$output" | grep "Train MSE:" | sed -n 's/.*Train MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    local trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)
    
    # 处理空值和NaN
    [ -z "$test_mse" ] || [ "$test_mse" = "-nan" ] && test_mse="ERROR"
    [ -z "$train_mse" ] || [ "$train_mse" = "-nan" ] && train_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"
    
    echo "$test_mse,$train_mse,$train_time,$trees"
}

# 运行单个测试
run_test() {
    local desc="$1"
    local objective="$2"
    local num_rounds="$3"
    local eta="$4"
    local max_depth="$5"
    local min_child_weight="$6"
    local lambda="$7"
    local gamma="$8"
    local subsample="$9"
    local colsample_bytree="${10}"
    
    local start_time=$(date +%s%3N)
    local output_text
    
    # 执行命令并捕获输出
    output_text=$($EXECUTABLE --data "$DATA_PATH" \
                         --objective "$objective" \
                         --num-rounds $num_rounds \
                         --eta $eta \
                         --max-depth $max_depth \
                         --min-child-weight $min_child_weight \
                         --lambda $lambda \
                         --gamma $gamma \
                         --subsample $subsample \
                         --colsample-bytree $colsample_bytree \
                         --quiet 2>&1)
    local exit_code=$?
    
    local end_time=$(date +%s%3N)
    local wall_time=$((end_time - start_time))
    
    # 如果程序执行失败，输出错误信息
    if [ $exit_code -ne 0 ]; then
        local error_line="%-25s | %-8s | %-8s | %-8s | %-5s | %s [ERROR: exit code %d]"
        # 控制台输出详细信息
        printf "$error_line\n" \
               "$desc" "ERROR" "0ms" "${wall_time}ms" "0" \
               "$objective $num_rounds $eta $max_depth $min_child_weight $lambda $gamma $subsample $colsample_bytree" \
               "$exit_code"
        # 文件输出简洁信息
        printf "%-25s | %-8s | %-8s | %-8s | ERROR\n" \
               "$desc" "ERROR" "0ms" "${wall_time}ms" >> "$RESULTS_FILE"
        return
    fi
    
    local metrics=$(extract_metrics "$output_text")
    local test_mse=$(echo "$metrics" | cut -d',' -f1)
    local train_mse=$(echo "$metrics" | cut -d',' -f2)
    local train_time=$(echo "$metrics" | cut -d',' -f3)
    local trees=$(echo "$metrics" | cut -d',' -f4)
    
    # 控制台输出详细信息
    printf "%-25s | %-8s | %-8s | %-8s | %-5s | %s\n" \
           "$desc" "$test_mse" "${train_time}ms" "${wall_time}ms" "$trees" \
           "$objective $num_rounds $eta $max_depth $min_child_weight $lambda $gamma $subsample $colsample_bytree"
    
    # 文件输出简洁信息
    printf "%-25s | %-8s | %-8s | %-8s | %-5s\n" \
           "$desc" "$test_mse" "${train_time}ms" "${wall_time}ms" "$trees" >> "$RESULTS_FILE"
}

# 打印表头
# 控制台输出详细表头
printf "%-25s | %-8s | %-8s | %-8s | %-5s | %s\n" "测试描述" "TestMSE" "TrainTime" "WallTime" "Trees" "参数配置"
echo "---------------------------------------------------------------------------------------------------------------"

# 文件输出简洁表头
printf "%-25s | %-8s | %-8s | %-8s | %-5s\n" "测试描述" "TestMSE" "TrainTime" "WallTime" "Trees" >> "$RESULTS_FILE"
echo "-------------------------------------------------------------------------" >> "$RESULTS_FILE"

output "=== 1. 目标函数测试 ==="
run_test "回归平方误差(基准)" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "回归线性(别名)" "reg:linear" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "回归逻辑(修复)" "reg:logistic" 30 0.3 6 1 1.0 0.0 1.0 1.0

output ""
output "=== 2. 迭代次数测试 ==="
run_test "迭代10次" "reg:squarederror" 10 0.3 6 1 1.0 0.0 1.0 1.0
run_test "迭代25次" "reg:squarederror" 25 0.3 6 1 1.0 0.0 1.0 1.0
run_test "迭代50次" "reg:squarederror" 50 0.3 6 1 1.0 0.0 1.0 1.0
run_test "迭代100次" "reg:squarederror" 100 0.3 6 1 1.0 0.0 1.0 1.0
run_test "迭代200次" "reg:squarederror" 200 0.3 6 1 1.0 0.0 1.0 1.0

output ""
output "=== 3. 学习率测试 ==="
run_test "学习率0.01" "reg:squarederror" 30 0.01 6 1 1.0 0.0 1.0 1.0
run_test "学习率0.05" "reg:squarederror" 30 0.05 6 1 1.0 0.0 1.0 1.0
run_test "学习率0.1" "reg:squarederror" 30 0.1 6 1 1.0 0.0 1.0 1.0
run_test "学习率0.3" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "学习率0.5" "reg:squarederror" 30 0.5 6 1 1.0 0.0 1.0 1.0

output ""
output "=== 4. 最大深度测试 ==="
run_test "深度2" "reg:squarederror" 30 0.3 2 1 1.0 0.0 1.0 1.0
run_test "深度3" "reg:squarederror" 30 0.3 3 1 1.0 0.0 1.0 1.0
run_test "深度4" "reg:squarederror" 30 0.3 4 1 1.0 0.0 1.0 1.0
run_test "深度6" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "深度8" "reg:squarederror" 30 0.3 8 1 1.0 0.0 1.0 1.0
run_test "深度10" "reg:squarederror" 30 0.3 10 1 1.0 0.0 1.0 1.0

output ""
output "=== 5. 最小子节点权重测试 ==="
run_test "最小权重1" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "最小权重2" "reg:squarederror" 30 0.3 6 2 1.0 0.0 1.0 1.0
run_test "最小权重5" "reg:squarederror" 30 0.3 6 5 1.0 0.0 1.0 1.0
run_test "最小权重10" "reg:squarederror" 30 0.3 6 10 1.0 0.0 1.0 1.0

output ""
output "=== 6. L2正则化参数测试 ==="
run_test "Lambda 0.1" "reg:squarederror" 30 0.3 6 1 0.1 0.0 1.0 1.0
run_test "Lambda 0.5" "reg:squarederror" 30 0.3 6 1 0.5 0.0 1.0 1.0
run_test "Lambda 1.0" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "Lambda 2.0" "reg:squarederror" 30 0.3 6 1 2.0 0.0 1.0 1.0
run_test "Lambda 5.0" "reg:squarederror" 30 0.3 6 1 5.0 0.0 1.0 1.0
run_test "Lambda 10.0" "reg:squarederror" 30 0.3 6 1 10.0 0.0 1.0 1.0

output ""
output "=== 7. 最小分裂损失测试 ==="
run_test "Gamma 0.0" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0
run_test "Gamma 0.1" "reg:squarederror" 30 0.3 6 1 1.0 0.1 1.0 1.0
run_test "Gamma 0.5" "reg:squarederror" 30 0.3 6 1 1.0 0.5 1.0 1.0
run_test "Gamma 1.0" "reg:squarederror" 30 0.3 6 1 1.0 1.0 1.0 1.0
run_test "Gamma 2.0" "reg:squarederror" 30 0.3 6 1 1.0 2.0 1.0 1.0

output ""
output "=== 8. 子采样率测试 ==="
run_test "子采样50%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 0.5 1.0
run_test "子采样70%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 0.7 1.0
run_test "子采样80%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 0.8 1.0
run_test "子采样90%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 0.9 1.0
run_test "无子采样" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0

output ""
output "=== 9. 列采样率测试 ==="
run_test "列采样50%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 0.5
run_test "列采样70%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 0.7
run_test "列采样80%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 0.8
run_test "列采样90%" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 0.9
run_test "无列采样" "reg:squarederror" 30 0.3 6 1 1.0 0.0 1.0 1.0

output ""
output "=== 10. 组合优化测试 ==="
run_test "保守配置" "reg:squarederror" 50 0.1 4 5 5.0 1.0 0.8 0.8
run_test "平衡配置" "reg:squarederror" 100 0.3 6 1 1.0 0.0 1.0 1.0
run_test "激进配置" "reg:squarederror" 200 0.5 8 1 0.1 0.0 1.0 1.0
run_test "高正则化" "reg:squarederror" 30 0.1 3 10 10.0 2.0 0.5 0.5
run_test "快速训练" "reg:squarederror" 20 0.5 3 1 0.5 0.0 1.0 1.0
run_test "深度拟合" "reg:squarederror" 100 0.1 10 1 2.0 0.5 0.9 0.9

# 记录结束时间
END_TIME=$(date)

output ""
output "=== XGBoost 全面测试完成 ==="
output "结束时间: $END_TIME"
output "结果已保存到: $RESULTS_FILE"

# 简单的调试测试 - 只在控制台显示
output_to_console_only ""
output_to_console_only "=== 调试信息 ==="
output_to_console_only "直接运行一个简单测试:"
$EXECUTABLE --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 10 --eta 0.3 --max-depth 3 --min-child-weight 1 --lambda 1.0 --gamma 0.0 --subsample 1.0 --colsample-bytree 1.0