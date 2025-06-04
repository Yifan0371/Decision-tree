#!/bin/bash

# =============================================================================
# script/boosting/gbrt/test_gbrt_parallel.sh
# GBRT 并行性能全面测试脚本
# 测试不同核数下的性能表现，包括不同配置的对比
# =============================================================================

# 项目根路径和可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# source env_config.sh 获取物理核数并自动构建
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"

# 创建结果目录
RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt"
mkdir -p "$RESULTS_DIR"

# 结果文件
RESULTS_FILE="$RESULTS_DIR/gbrt_parallel_performance_results.txt"
> "$RESULTS_FILE"

echo "=========================================="
echo "    GBRT Parallel Performance Test     "
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
    echo "# GBRT Parallel Performance Test Results"
    echo "# Date: $(date)"
    echo "# Max Cores: $MAX_CORES"
    echo "# Data: $(basename $DATA_PATH)"
    echo "# Format: Config | Threads | TestMSE | TrainTime(ms) | TotalTime(ms) | Speedup | Efficiency"
} >> "$RESULTS_FILE"

# 解析输出的函数
extract_results() {
    local output="$1"
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    local trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)
    
    # 处理空值
    [ -z "$test_mse" ] && test_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"
    
    echo "$test_mse,$train_time,$trees"
}

# 运行单个测试
run_test() {
    local config_name="$1"
    local threads="$2"
    local loss="$3"
    local iterations="$4" 
    local learning_rate="$5"
    local max_depth="$6"
    local min_leaf="$7"
    local criterion="$8"
    local split_method="$9"
    local subsample="${10}"
    
    export OMP_NUM_THREADS=$threads
    
    echo -n "  测试 $config_name (${threads}线程)... "
    
    local start_time=$(date +%s%3N)
    local output
    output=$($EXECUTABLE "$DATA_PATH" \
        "$loss" $iterations $learning_rate $max_depth $min_leaf \
        "$criterion" "$split_method" $subsample 2>&1)
    local exit_code=$?
    local end_time=$(date +%s%3N)
    local wall_time=$((end_time - start_time))
    
    if [ $exit_code -ne 0 ]; then
        echo "FAILED"
        echo "$config_name,$threads,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
        return
    fi
    
    local results=$(extract_results "$output")
    local test_mse=$(echo "$results" | cut -d',' -f1)
    local train_time=$(echo "$results" | cut -d',' -f2)
    local trees=$(echo "$results" | cut -d',' -f3)
    
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
    echo "$config_name,$threads,$test_mse,$train_time,$wall_time,$speedup,$efficiency" >> "$RESULTS_FILE"
}

# 测试配置定义
declare -A test_configs
declare -A baseline_times

test_configs["FastSmall"]="squared 20 0.1 3 2 mse exhaustive 1.0"
test_configs["Standard"]="squared 30 0.1 4 1 mse exhaustive 1.0" 
test_configs["DeepTrees"]="squared 25 0.1 6 1 mse exhaustive 1.0"
test_configs["RandomSplit"]="squared 30 0.1 4 1 mse random 1.0"
test_configs["HistogramSplit"]="squared 30 0.1 4 1 mse histogram_ew 1.0"
test_configs["HuberLoss"]="huber 30 0.1 4 1 mse exhaustive 1.0"
test_configs["Subsample"]="squared 30 0.1 4 1 mse exhaustive 0.8"

echo "开始性能测试..."
echo ""

# 主测试循环
for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit" "HuberLoss" "Subsample"; do
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
    for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit" "HuberLoss" "Subsample"; do
        echo "=== $config_name ==="
        echo "Threads | TestMSE    | TrainTime  | Speedup | Efficiency"
        echo "--------|------------|------------|---------|----------"
        
        grep "^$config_name," "$RESULTS_FILE" | while IFS=',' read -r cfg threads mse train_time wall_time speedup efficiency; do
            printf "%-7s | %-10s | %-10s | %-7s | %s\n" "$threads" "$mse" "${train_time}ms" "$speedup" "$efficiency"
        done
        echo ""
    done
    
    echo "===== PARALLEL SCALING ANALYSIS ====="
    echo ""
    echo "理想并行性能指标:"
    echo "- 加速比 (Speedup): 接近线程数表示良好的强扩展性"
    echo "- 效率 (Efficiency): 接近1.0表示线程利用率高"
    echo "- 线性扩展: 效率 > 0.8 通常认为是良好的"
    echo ""
    
    echo "GBRT并行特性分析:"
    echo "- 梯度计算：高度并行，效率接近1.0"
    echo "- 树构建：受分割查找影响，中等并行度"
    echo "- 预测更新：高度并行，效率接近1.0"
    echo "- 整体：预期效率0.7-0.9，取决于数据大小和树深度"
    echo ""
    
} >> "$RESULTS_FILE"

echo "详细结果已保存到: $RESULTS_FILE"
echo ""

# 显示简要汇总
echo "=== 简要性能汇总 ==="
printf "%-15s | %-7s | %-10s | %-7s | %s\n" "配置" "线程数" "训练时间" "加速比" "效率"
echo "----------------|---------|------------|---------|----------"

for config_name in "Standard" "DeepTrees" "RandomSplit"; do
    # 显示1核和最大核数的对比
    baseline_result=$(grep "^$config_name,1," "$RESULTS_FILE" | cut -d',' -f4)
    maxcore_line=$(grep "^$config_name,$MAX_CORES," "$RESULTS_FILE")
    
    if [ -n "$maxcore_line" ]; then
        maxcore_time=$(echo "$maxcore_line" | cut -d',' -f4)
        maxcore_speedup=$(echo "$maxcore_line" | cut -d',' -f6)
        maxcore_efficiency=$(echo "$maxcore_line" | cut -d',' -f7)
        
        printf "%-15s | %-7s | %-10s | %-7s | %s\n" \
               "$config_name" "1" "${baseline_result}ms" "1.00" "1.00"
        printf "%-15s | %-7s | %-10s | %-7s | %s\n" \
               "" "$MAX_CORES" "${maxcore_time}ms" "$maxcore_speedup" "$maxcore_efficiency"
        echo "----------------|---------|------------|---------|----------"
    fi
done

echo ""
echo "GBRT并行测试完成！"
echo "详细结果: $RESULTS_FILE"
echo "建议查看完整报告以分析不同配置的并行扩展性。"