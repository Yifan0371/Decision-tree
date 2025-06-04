#!/bin/bash

# =============================================================================
# script/boosting/xgboost/test_xgboost_parallel.sh
# XGBoost 并行性能全面测试脚本
# 测试不同核数下的性能表现，包括不同配置的对比
# =============================================================================

# 项目根路径和可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# source env_config.sh 获取物理核数并自动构建
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/XGBoostMain"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"

# 创建结果目录
RESULTS_DIR="$PROJECT_ROOT/script/boosting/xgboost"
mkdir -p "$RESULTS_DIR"

# 结果文件
RESULTS_FILE="$RESULTS_DIR/xgboost_parallel_performance_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "    XGBoost Parallel Performance Test     "
echo "=========================================="
echo "物理核数: $MAX_CORES"
echo "数据文件: $(basename $DATA_PATH)"
echo "结果保存: $RESULTS_FILE"
echo "时间: $(date)"
echo ""

# 检查文件存在性
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: $EXECUTABLE not found!"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file $DATA_PATH not found!"
    # 尝试其他数据文件
    for alt_data in "$PROJECT_ROOT/data/data_clean/cleaned_sample_400_rows.csv" "$PROJECT_ROOT/data/data_clean/cleaned_15k_random.csv"; do
        if [ -f "$alt_data" ]; then
            echo "使用替代数据文件: $(basename $alt_data)"
            DATA_PATH="$alt_data"
            break
        fi
    done
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "未找到任何可用的数据文件"
        exit 1
    fi
fi

# 生成线程数列表
threads_list=(1)
current=1
while (( current * 2 <= MAX_CORES )); do
    current=$((current * 2))
    threads_list+=($current)
done
# 如果最大值不等于MAX_CORES，添加MAX_CORES
if (( current != MAX_CORES )); then
    threads_list+=($MAX_CORES)
fi

echo "测试线程数序列: ${threads_list[*]}"
echo ""

# 写入结果文件头部
{
    echo "# XGBoost Parallel Performance Test Results"
    echo "# Date: $(date)"
    echo "# Max Cores: $MAX_CORES"
    echo "# Data: $(basename $DATA_PATH)"
    echo "# Format: Config | Threads | TestMSE | TestMAE | TrainTime(ms) | TotalTime(ms) | Speedup | Efficiency"
} > "$RESULTS_FILE"

# 解析输出的函数
extract_results() {
    local output="$1"
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local test_mae=$(echo "$output" | grep "Test MAE:" | sed -n 's/.*Test MAE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    
    # 处理空值
    [ -z "$test_mse" ] && test_mse="ERROR"
    [ -z "$test_mae" ] && test_mae="ERROR" 
    [ -z "$train_time" ] && train_time="0"
    
    echo "$test_mse,$test_mae,$train_time"
}

# 运行单个测试
run_test() {
    local config_name="$1"
    local threads="$2"
    local objective="$3"
    local num_rounds="$4"
    local eta="$5"
    local max_depth="$6"
    local min_child_weight="$7"
    local lambda="$8"
    local gamma="$9"
    local subsample="${10}"
    local colsample_bytree="${11}"
    
    export OMP_NUM_THREADS=$threads
    
    echo -n "  测试 $config_name (${threads}线程)... "
    
    local start_time=$(date +%s%3N)
    local output
    output=$($EXECUTABLE --data "$DATA_PATH" \
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
    
    if [ $exit_code -ne 0 ]; then
        echo "FAILED"
        echo "$config_name,$threads,ERROR,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
        return
    fi
    
    local results=$(extract_results "$output")
    local test_mse=$(echo "$results" | cut -d',' -f1)
    local test_mae=$(echo "$results" | cut -d',' -f2)
    local train_time=$(echo "$results" | cut -d',' -f3)
    
    echo "完成 (${train_time}ms)"
    
    # 计算加速比和效率
    local speedup="N/A"
    local efficiency="N/A"
    if [ "$threads" -eq 1 ]; then
        # 存储基准时间
        baseline_times["$config_name"]="$train_time"
        speedup="1.00"
        efficiency="1.00"
    else
        # 获取基准时间
        local baseline_time="${baseline_times[$config_name]:-0}"
        if [ "$baseline_time" -gt 0 ] && [ "$train_time" -gt 0 ]; then
            speedup=$(echo "scale=2; $baseline_time / $train_time" | bc -l 2>/dev/null || echo "N/A")
            efficiency=$(echo "scale=2; $speedup / $threads" | bc -l 2>/dev/null || echo "N/A")
        fi
    fi
    
    # 写入结果
    echo "$config_name,$threads,$test_mse,$test_mae,$train_time,$wall_time,$speedup,$efficiency" >> "$RESULTS_FILE"
}

# 测试配置定义
declare -A test_configs
declare -A baseline_times

# 配置格式: objective num_rounds eta max_depth min_child_weight lambda gamma subsample colsample_bytree
test_configs["FastSmall"]="reg:squarederror 50 0.3 4 1 1.0 0.0 1.0 1.0"
test_configs["Standard"]="reg:squarederror 100 0.3 6 1 1.0 0.0 1.0 1.0" 
test_configs["DeepTrees"]="reg:squarederror 80 0.1 8 1 2.0 0.5 1.0 1.0"
test_configs["HighRegularization"]="reg:squarederror 100 0.1 4 5 10.0 2.0 0.8 0.8"
test_configs["FastTraining"]="reg:squarederror 150 0.5 3 1 0.1 0.0 1.0 1.0"

echo "开始性能测试..."
echo ""

# 主测试循环
for config_name in "FastSmall" "Standard" "DeepTrees" "HighRegularization" "FastTraining"; do
    echo "=== 配置: $config_name ==="
    
    # 解析配置参数
    IFS=' ' read -ra params <<< "${test_configs[$config_name]}"
    
    for threads in "${threads_list[@]}"; do
        run_test "$config_name" "$threads" "${params[@]}"
    done
    
    echo ""
done

echo "=========================================="
echo "测试完成！生成结果汇总..."
echo ""

# 生成汇总报告
{
    echo ""
    echo "===== PERFORMANCE SUMMARY ====="
    echo ""
    
    # 按配置分组显示结果
    for config_name in "FastSmall" "Standard" "DeepTrees" "HighRegularization" "FastTraining"; do
        echo "=== $config_name ==="
        echo "Threads | TestMSE    | TrainTime  | Speedup | Efficiency"
        echo "--------|------------|------------|---------|----------"
        
        grep "^$config_name," "$RESULTS_FILE" | while IFS=',' read -r cfg threads mse mae train_time wall_time speedup efficiency; do
            printf "%-7s | %-10s | %-10s | %-7s | %s\n" "$threads" "$mse" "${train_time}ms" "$speedup" "$efficiency"
        done
        echo ""
    done
    
    echo "===== XGBOOST PARALLEL SCALING ANALYSIS ====="
    echo ""
    echo "理想并行性能指标:"
    echo "- 加速比 (Speedup): 接近线程数表示良好的强扩展性"
    echo "- 效率 (Efficiency): 接近1.0表示线程利用率高"
    echo "- 线性扩展: 效率 > 0.8 通常认为是良好的"
    echo "- 超线性扩展: 加速比 > 线程数（由于缓存效应）"
    echo ""
    
    echo "XGBoost性能优化建议:"
    echo "- 如果效率随线程数快速下降，考虑减少树深度或增加数据量"
    echo "- 深度树配置在多核上的扩展性可能不如浅树"
    echo "- 高正则化配置通常具有更好的并行扩展性"
    echo "- 对于小数据集，建议使用较少的线程数"
    echo ""
    
} >> "$RESULTS_FILE"

echo "详细结果已保存到: $RESULTS_FILE"
echo ""

# 显示简要汇总
echo "=== 简要性能汇总 ==="
printf "%-20s | %-7s | %-10s | %-7s | %s\n" "配置" "线程数" "训练时间" "加速比" "效率"
echo "--------------------|---------|------------|---------|----------"

for config_name in "FastSmall" "Standard" "DeepTrees" "HighRegularization" "FastTraining"; do
    # 显示1核和最大核数的对比
    baseline_result=$(grep "^$config_name,1," "$RESULTS_FILE" | cut -d',' -f5)
    maxcore_line=$(grep "^$config_name,$MAX_CORES," "$RESULTS_FILE")
    
    if [ -n "$maxcore_line" ]; then
        maxcore_time=$(echo "$maxcore_line" | cut -d',' -f5)
        maxcore_speedup=$(echo "$maxcore_line" | cut -d',' -f7)
        maxcore_efficiency=$(echo "$maxcore_line" | cut -d',' -f8)
        
        printf "%-20s | %-7s | %-10s | %-7s | %s\n" \
               "$config_name" "1" "${baseline_result}ms" "1.00" "1.00"
        printf "%-20s | %-7s | %-10s | %-7s | %s\n" \
               "" "$MAX_CORES" "${maxcore_time}ms" "$maxcore_speedup" "$maxcore_efficiency"
        echo "--------------------|---------|------------|---------|----------"
    fi
done

echo ""
echo "XGBoost并行测试完成！"
echo "详细结果: $RESULTS_FILE"
echo "建议查看完整报告以分析不同配置的并行扩展性。"