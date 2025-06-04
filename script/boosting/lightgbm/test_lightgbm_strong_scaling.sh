#!/bin/bash
set -euo pipefail

# ==============================================================================
# LightGBM Strong-scaling test – 固定数据规模，按线程数测试耗时和 MSE，
# 并将结果保存到日志文件
# ==============================================================================

# 1) 定位项目根目录 & 日志目录
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/lightgbm"
mkdir -p "$SCRIPT_DIR"

# 定义并创建日志文件（用时间戳区分每次运行）
LOGFILE="$SCRIPT_DIR/lightgbm_strong_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "LightGBM Strong Scaling Performance Test Log" >> "$LOGFILE"
echo "运行时间: $(date)" >> "$LOGFILE"
echo "项目根: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# 将后续所有输出同时写入终端和日志文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/LightGBMMain"

# 2) source env_config.sh 自动设置 OMP_NUM_THREADS，并在必要时编译
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
NUM_ITERATIONS=100
LEARNING_RATE=0.1
NUM_LEAVES=31
MIN_DATA_IN_LEAF=20
SPLIT_METHOD="histogram_eq:64"

# 4) 确认可执行和数据文件存在
[[ -f "$EXECUTABLE" ]] || { echo " $EXECUTABLE 不存在"; exit 1; }
[[ -f "$DATA"       ]] || { echo " $DATA 不存在"; exit 1; }

# 5) 生成线程列表：1, 2, 4, …, MAX_CORES，如果最后一个不是 MAX_CORES 再加一个
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
    threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 6) 打印表头
echo "==============================================="
echo "    LightGBM Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Iterations: $NUM_ITERATIONS | Learning Rate: $LEARNING_RATE"
echo "  Num Leaves: $NUM_LEAVES | Min Data in Leaf: $MIN_DATA_IN_LEAF"
echo "  Split Method: $SPLIT_METHOD"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TestMAE    | Trees      | LGB/sec"
echo "--------|-------------|------------|------------|------------|----------"

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) 开始计时
    start_ts=$(date +%s%3N)

    # 6.2) 执行 LightGBM 并捕获完整 stdout
    output=$("$EXECUTABLE" --data "$DATA" \
        --split-method "$SPLIT_METHOD" \
        --num-iterations $NUM_ITERATIONS \
        --learning-rate $LEARNING_RATE \
        --num-leaves $NUM_LEAVES \
        --min-data-in-leaf $MIN_DATA_IN_LEAF \
        --quiet 2>/dev/null)

    # 6.3) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 6.4) 提取关键指标
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(echo "$output" | grep "Test MAE:" | sed -n 's/.*Test MAE: \([0-9.-]*\).*/\1/p' | tail -1)
    trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

    # 计算吞吐量：迭代/秒
    lgb_per_sec=$(echo "scale=2; $NUM_ITERATIONS * 1000 / $elapsed" | bc -l 2>/dev/null || echo "N/A")

    # 处理空值
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    [[ -z "$trees" ]] && trees="ERROR"

    # 6.5) 打印并记录本次结果行
    printf "%7d | %11d | %-10s | %-10s | %-10s | %s\n" \
           "$t" "$elapsed" "$test_mse" "$test_mae" "$trees" "$lgb_per_sec"
done

echo ""
echo "==============================================="
echo "Strong Scaling Analysis:"
echo "- 理想: 线性加速，时间反比于线程数"
echo "- 关注点: TestMSE 保持稳定，Elapsed 时间下降"
echo "- 效率 = (串行时间 / 并行时间) / 线程数"
echo "- LightGBM 特有: GOSS和EFB优化对并行性能的影响"
echo "==============================================="

exit 0