#!/bin/bash
set -euo pipefail

# ==============================================================================
# LightGBM Weak-scaling test（自动按行截取同一大文件）– 对同一份数据，
# 截取不同大小子集，按线程数测试耗时和 MSE，并将结果保存到日志文件
# ==============================================================================

# 1) 定位项目根目录和可执行
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/lightgbm"
mkdir -p "$SCRIPT_DIR"

# 定义并创建日志文件（用时间戳区分每次运行）
LOGFILE="$SCRIPT_DIR/lightgbm_weak_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "LightGBM Weak Scaling Performance Test Log" >> "$LOGFILE"
echo "运行时间: $(date)" >> "$LOGFILE"
echo "项目根: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# 将后续所有输出同时写入终端和日志文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/LightGBMMain"

# 2) source env_config.sh：自动设置 OMP_NUM_THREADS，并在必要时编译可执行
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定使用的原始数据文件：cleaned_data.csv
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
NUM_ITERATIONS=100
LEARNING_RATE=0.1
NUM_LEAVES=31
MIN_DATA_IN_LEAF=20
SPLIT_METHOD="histogram_eq:64"

# 4) 确认可执行和数据都存在
[[ -f "$EXECUTABLE" ]] || { echo " $EXECUTABLE 不存在"; exit 1; }
[[ -f "$DATA"       ]] || { echo " $DATA 不存在"; exit 1; }

# 5) 计算原始数据的总行数（不含 header）和基准行数 BASE
total_rows=$(( $(wc -l < "$DATA") - 1 ))
if (( total_rows < MAX_CORES )); then
    echo "警告：数据行数 ($total_rows) 小于物理核数 ($MAX_CORES)，脚本退出"
    exit 1
fi
BASE=$(( total_rows / MAX_CORES ))
echo "总行数 (不含 header): $total_rows，物理核数: $MAX_CORES，基准行数 BASE=$BASE"

# 6) 生成线程列表：1, 2, 4, …, MAX_CORES。如果最后一个不是 MAX_CORES 则再加上
threads=(1)
while (( threads[-1] * 2 <= MAX_CORES )); do
    threads+=( $(( threads[-1] * 2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 7) 打印结果表头
echo "==============================================="
echo "    LightGBM Weak Scaling Performance Test     "
echo "==============================================="
echo "Fixed Parameters (per thread):"
echo "  Iterations: $NUM_ITERATIONS | Learning Rate: $LEARNING_RATE"
echo "  Num Leaves: $NUM_LEAVES | Min Data in Leaf: $MIN_DATA_IN_LEAF"
echo "  Split Method: $SPLIT_METHOD"
echo "  Base rows per thread: $BASE"
echo ""
echo "Threads | SubsetRows | Elapsed(ms) | TestMSE    | TestMAE    | Trees      | Efficiency"
echo "--------|------------|-------------|------------|------------|------------|----------"

# 记录单线程时间作为基准
baseline_time=0

# 8) 对于每个线程数 t：
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 8.1) 计算本次需截取的样本行数 chunk_size = t * BASE
    chunk_size=$(( t * BASE ))
    # 如果 chunk_size 超过了 total_rows，就直接取全部
    if (( chunk_size > total_rows )); then
        chunk_size=$total_rows
    fi
    # head 需要截取的行数 = 表头1行 + chunk_size行
    lines_to_take=$(( chunk_size + 1 ))

    # 8.2) 生成一个临时文件 tmpfile，内容为 header + 前 lines_to_take-1 条样本
    tmpfile="$PROJECT_ROOT/data/data_clean/tmp_lightgbm_chunk_t${t}.csv"
    head -n "$lines_to_take" "$DATA" > "$tmpfile"

    # 8.3) 开始计时
    start_ts=$(date +%s%3N)

    # 8.4) 运行 LightGBM，并捕获完整 stdout
    output=$("$EXECUTABLE" --data "$tmpfile" \
        --split-method "$SPLIT_METHOD" \
        --num-iterations $NUM_ITERATIONS \
        --learning-rate $LEARNING_RATE \
        --num-leaves $NUM_LEAVES \
        --min-data-in-leaf $MIN_DATA_IN_LEAF \
        --quiet 2>/dev/null)

    # 8.5) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 8.6) 提取关键指标
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(echo "$output" | grep "Test MAE:" | sed -n 's/.*Test MAE: \([0-9.-]*\).*/\1/p' | tail -1)
    trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

    # 处理空值
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    [[ -z "$trees" ]] && trees="ERROR"

    # 8.7) 计算效率 (相对于理想 weak scaling 的效率)
    if (( t == 1 )); then
        baseline_time=$elapsed
        efficiency="1.00"
    else
        if (( elapsed > 0 )); then
            efficiency=$(echo "scale=2; $baseline_time / $elapsed" | bc -l 2>/dev/null || echo "N/A")
        else
            efficiency="N/A"
        fi
    fi

    # 8.8) 打印并记录本次结果行
    printf "%7d | %10d | %11d | %-10s | %-10s | %-10s | %s\n" \
           "$t" "$chunk_size" "$elapsed" "$test_mse" "$test_mae" "$trees" "$efficiency"

    # 8.9) 删除临时文件
    rm -f "$tmpfile"
done

echo ""
echo "==============================================="
echo "Weak Scaling Analysis:"
echo "- 理想: 随着线程数增加，处理更多数据但时间保持恒定"
echo "- 效率 = 单线程时间 / 当前时间"
echo "- 关注点: 效率接近1.0表示良好的 weak scaling"
echo "- 数据量按线程数线性增长，计算复杂度也线性增长"
echo "- LightGBM 特点: GOSS采样对大数据集的扩展性影响"
echo "==============================================="

exit 0