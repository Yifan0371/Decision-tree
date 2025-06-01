#!/bin/bash

# =============================================================================
# script/boosting/compare_boosting_methods_fixed.sh
# XGBoost vs GBRT 性能对比脚本 (修复MSE提取问题)
# 在项目根目录运行: bash script/boosting/compare_boosting_methods_fixed.sh
# =============================================================================

DATA_PATH="data/data_clean/cleaned_data.csv"
XGBOOST_EXECUTABLE="build/XGBoostMain"
GBRT_EXECUTABLE="build/RegressionBoostingMain"

# 结果文件路径
RESULTS_DIR="script/boosting"
RESULTS_FILE="$RESULTS_DIR/boosting_comparison_results_fixed.txt"

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

output "=================================================="
output "           Boosting Methods Comparison           "
output "         XGBoost vs GBRT Performance (Fixed)     "
output "=================================================="
output "开始时间: $START_TIME"
output "数据文件: $DATA_PATH"
output "XGBoost可执行文件: $XGBOOST_EXECUTABLE"
output "GBRT可执行文件: $GBRT_EXECUTABLE"
output "结果保存到: $RESULTS_FILE"
output "修复说明: 使用精确的MSE提取方法"
output ""

# 检查文件是否存在
missing_files=""
if [ ! -f "$XGBOOST_EXECUTABLE" ]; then
    missing_files="$missing_files $XGBOOST_EXECUTABLE"
fi

if [ ! -f "$GBRT_EXECUTABLE" ]; then
    missing_files="$missing_files $GBRT_EXECUTABLE"
fi

if [ ! -f "$DATA_PATH" ]; then
    missing_files="$missing_files $DATA_PATH"
fi

if [ -n "$missing_files" ]; then
    output "错误: 找不到以下文件:$missing_files"
    output "请确保已编译所有可执行文件: cd build && make"
    
    # 尝试其他可能的数据文件
    if [ ! -f "$DATA_PATH" ]; then
        for alt_data in "data/data_clean/cleaned_sample_100_rows.csv" "data/data_clean/cleaned_sample_400_rows.csv" "data/data_clean/cleaned_15k_random.csv"; do
            if [ -f "$alt_data" ]; then
                output "使用替代数据文件: $alt_data"
                DATA_PATH="$alt_data"
                break
            fi
        done
    fi
    
    if [ ! -f "$DATA_PATH" ]; then
        exit 1
    fi
fi

# 修复的XGBoost结果解析函数
extract_xgboost_metrics() {
    local output="$1"
    # 使用精确的sed匹配，避免匹配配置参数中的数字
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_mse=$(echo "$output" | grep "Train MSE:" | sed -n 's/.*Train MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    local trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)
    
    # 处理空值
    [ -z "$test_mse" ] && test_mse="ERROR"
    [ -z "$train_mse" ] && train_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"
    
    echo "$test_mse,$train_mse,$train_time,$trees"
}

# 修复的GBRT结果解析函数
extract_gbrt_metrics() {
    local output="$1"
    # 使用精确的sed匹配
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_mse=$(echo "$output" | grep "Train MSE:" | sed -n 's/.*Train MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    local trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)
    
    # 处理空值
    [ -z "$test_mse" ] && test_mse="ERROR" 
    [ -z "$train_mse" ] && train_mse="ERROR" 
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"
    
    echo "$test_mse,$train_mse,$train_time,$trees"
}

# 运行对比测试
run_comparison() {
    local desc="$1"
    local num_rounds="$2"
    local eta="$3"
    local max_depth="$4"
    local min_samples="$5"
    local subsample="$6"
    local lambda="${7:-1.0}"
    local gamma="${8:-0.0}"
    
    output "测试配置: $desc"
    output "参数: rounds=$num_rounds, eta=$eta, depth=$max_depth, min_samples=$min_samples, subsample=$subsample"
    
    # 运行XGBoost
    output "  运行XGBoost..."
    local xgb_start=$(date +%s%3N)
    local xgb_output=$($XGBOOST_EXECUTABLE --data "$DATA_PATH" \
                                          --objective "reg:squarederror" \
                                          --num-rounds $num_rounds \
                                          --eta $eta \
                                          --max-depth $max_depth \
                                          --min-child-weight $min_samples \
                                          --lambda $lambda \
                                          --gamma $gamma \
                                          --subsample $subsample \
                                          --colsample-bytree 1.0 \
                                          --quiet 2>&1)
    local xgb_exit_code=$?
    local xgb_end=$(date +%s%3N)
    local xgb_wall_time=$((xgb_end - xgb_start))
    
    # 运行GBRT
    output "  运行GBRT..."
    local gbrt_start=$(date +%s%3N)
    local gbrt_output=$($GBRT_EXECUTABLE "$DATA_PATH" \
                                        "squared" \
                                        $num_rounds \
                                        $eta \
                                        $max_depth \
                                        $min_samples \
                                        "mse" \
                                        "exhaustive" \
                                        $subsample 2>&1)
    local gbrt_exit_code=$?
    local gbrt_end=$(date +%s%3N)
    local gbrt_wall_time=$((gbrt_end - gbrt_start))
    
    # 输出结果
    output_to_console_only ""
    printf "%-15s | %-8s | %-8s | %-8s | %-5s | %-8s\n" "算法" "TestMSE" "TrainTime" "WallTime" "Trees" "状态"
    echo "--------------------------------------------------------------------"
    
    # 文件输出简洁表头
    output_to_file_only ""
    printf "%-15s | %-8s | %-8s | %-8s | %-5s\n" "算法" "TestMSE" "TrainTime" "WallTime" "Trees" >> "$RESULTS_FILE"
    echo "-------------------------------------------------------" >> "$RESULTS_FILE"
    
    if [ $xgb_exit_code -eq 0 ]; then
        local xgb_metrics=$(extract_xgboost_metrics "$xgb_output")
        local xgb_test_mse=$(echo "$xgb_metrics" | cut -d',' -f1)
        local xgb_train_time=$(echo "$xgb_metrics" | cut -d',' -f3)
        local xgb_trees=$(echo "$xgb_metrics" | cut -d',' -f4)
        
        # 控制台详细输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s | %-8s\n" \
               "XGBoost" "$xgb_test_mse" "${xgb_train_time}ms" "${xgb_wall_time}ms" "$xgb_trees" "成功"
        # 文件简洁输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s\n" \
               "XGBoost" "$xgb_test_mse" "${xgb_train_time}ms" "${xgb_wall_time}ms" "$xgb_trees" >> "$RESULTS_FILE"
    else
        # 控制台详细输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s | %-8s\n" \
               "XGBoost" "ERROR" "0ms" "${xgb_wall_time}ms" "0" "失败"
        # 文件简洁输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s\n" \
               "XGBoost" "ERROR" "0ms" "${xgb_wall_time}ms" "0" >> "$RESULTS_FILE"
    fi
    
    if [ $gbrt_exit_code -eq 0 ]; then
        local gbrt_metrics=$(extract_gbrt_metrics "$gbrt_output")
        local gbrt_test_mse=$(echo "$gbrt_metrics" | cut -d',' -f1)
        local gbrt_train_time=$(echo "$gbrt_metrics" | cut -d',' -f3)
        local gbrt_trees=$(echo "$gbrt_metrics" | cut -d',' -f4)
        
        # 控制台详细输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s | %-8s\n" \
               "GBRT" "$gbrt_test_mse" "${gbrt_train_time}ms" "${gbrt_wall_time}ms" "$gbrt_trees" "成功"
        # 文件简洁输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s\n" \
               "GBRT" "$gbrt_test_mse" "${gbrt_train_time}ms" "${gbrt_wall_time}ms" "$gbrt_trees" >> "$RESULTS_FILE"
    else
        # 控制台详细输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s | %-8s\n" \
               "GBRT" "ERROR" "0ms" "${gbrt_wall_time}ms" "0" "失败"
        # 文件简洁输出
        printf "%-15s | %-8s | %-8s | %-8s | %-5s\n" \
               "GBRT" "ERROR" "0ms" "${gbrt_wall_time}ms" "0" >> "$RESULTS_FILE"
    fi
    
    # 性能对比分析
    if [ $xgb_exit_code -eq 0 ] && [ $gbrt_exit_code -eq 0 ]; then
        output ""
        output "性能对比分析:"
        
        # 比较测试MSE
        if [ "$xgb_test_mse" != "ERROR" ] && [ "$gbrt_test_mse" != "ERROR" ]; then
            local mse_diff=$(echo "$xgb_test_mse $gbrt_test_mse" | awk '{printf "%.6f", $1 - $2}')
            local mse_ratio=$(echo "$xgb_test_mse $gbrt_test_mse" | awk '{if($2 != 0) printf "%.2f", $1 / $2; else print "N/A"}')
            
            if (( $(echo "$mse_diff < 0" | bc -l 2>/dev/null || echo "0") )); then
                output "  - XGBoost测试MSE更低: $xgb_test_mse vs $gbrt_test_mse (相对差异: ${mse_diff#-})"
            elif (( $(echo "$mse_diff > 0" | bc -l 2>/dev/null || echo "0") )); then
                output "  - GBRT测试MSE更低: $gbrt_test_mse vs $xgb_test_mse (相对差异: $mse_diff)"
            else
                output "  - 两者测试MSE相近: $xgb_test_mse vs $gbrt_test_mse"
            fi
            output "  - MSE比值: XGBoost/GBRT = $mse_ratio"
        fi
        
        # 比较训练时间
        if [ "$xgb_train_time" != "0" ] && [ "$gbrt_train_time" != "0" ]; then
            local time_ratio=$(echo "$xgb_train_time $gbrt_train_time" | awk '{printf "%.2f", $1 / $2}')
            if (( $(echo "$time_ratio < 1" | bc -l 2>/dev/null || echo "0") )); then
                output "  - XGBoost训练更快: ${xgb_train_time}ms vs ${gbrt_train_time}ms (${time_ratio}x)"
            else
                output "  - GBRT训练更快: ${gbrt_train_time}ms vs ${xgb_train_time}ms ($(echo "$time_ratio" | awk '{printf "%.2f", 1/$1}')x)"
            fi
        fi
        
        # 比较墙上时间
        if [ $xgb_wall_time -lt $gbrt_wall_time ]; then
            local speedup=$(echo "$gbrt_wall_time $xgb_wall_time" | awk '{printf "%.2f", $1 / $2}')
            output "  - XGBoost总时间更短: ${xgb_wall_time}ms vs ${gbrt_wall_time}ms (${speedup}x 加速)"
        else
            local speedup=$(echo "$xgb_wall_time $gbrt_wall_time" | awk '{printf "%.2f", $1 / $2}')
            output "  - GBRT总时间更短: ${gbrt_wall_time}ms vs ${xgb_wall_time}ms (${speedup}x 加速)"
        fi
        
        # MSE合理性检查
        if [ "$xgb_test_mse" != "ERROR" ] && [ "$gbrt_test_mse" != "ERROR" ]; then
            local xgb_magnitude=$(echo "$xgb_test_mse" | awk '{if($1 > 0.01) print "高"; else if($1 > 0.001) print "中"; else print "低"}')
            local gbrt_magnitude=$(echo "$gbrt_test_mse" | awk '{if($1 > 0.01) print "高"; else if($1 > 0.001) print "中"; else print "低"}')
            output "  - MSE数量级: XGBoost($xgb_magnitude), GBRT($gbrt_magnitude)"
        fi
    fi
    
    output ""
    output "========================================"
    output ""
}

# 执行对比测试
output "开始运行对比测试 (使用修复的MSE提取)..."
output ""

output "=== 1. 基准配置对比 ==="
run_comparison "基准配置" 30 0.1 4 1 1.0 1.0 0.0

output "=== 2. 高学习率对比 ==="
run_comparison "高学习率" 30 0.3 4 1 1.0 1.0 0.0

output "=== 3. 深树对比 ==="
run_comparison "深树配置" 30 0.1 8 1 1.0 1.0 0.0

# 记录结束时间
END_TIME=$(date)

output "=================================================="
output "                 测试总结                         "
output "=================================================="
output "结束时间: $END_TIME"
output "测试配置总数: 3 (快速验证)"
output "数据集: $DATA_PATH"
output "结果已保存到: $RESULTS_FILE"
output ""
output "修复验证:"
output "- 使用精确的sed匹配避免提取配置参数中的数字"
output "- XGBoost和GBRT的MSE应该在相同数量级"
output "- 预期MSE范围: 0.0002-0.0005"
output ""
output "=================================================="