#!/bin/bash
set -euo pipefail

# ==============================================================================
# script/boosting/gbrt/test_gbrt_strong_scaling.sh
# GBRT Strong-scaling test – 固定数据规模，按线程数测试耗时和MSE
# ==============================================================================

# 1) 定位项目根目录 & 日志目录
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/gbrt"
mkdir -p "$SCRIPT_DIR"

# 定义并创建日志文件（用时间戳区分每次运行）
LOGFILE="$SCRIPT_DIR/gbrt_strong_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "GBRT Strong Scaling Performance Test Log" >> "$LOGFILE"
echo "运行时间: $(date)" >> "$LOGFILE"
echo "项目根: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# 将后续所有输出同时写入终端和日志文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

# 2) source env_config.sh 自动设置 OMP_NUM_THREADS，并在必要时编译
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
LOSS="squared"
NUM_ITERATIONS=30
LEARNING_RATE=0.1
MAX_DEPTH=4
MIN_LEAF=1
CRITERION="mse"
SPLIT_METHOD="exhaustive"
SUBSAMPLE=1.0

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
echo "    GBRT Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Loss: $LOSS | Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LEARNING_RATE | Max Depth: $MAX_DEPTH"
echo "  Min Leaf: $MIN_LEAF | Criterion: $CRITERION"
echo "  Split Method: $SPLIT_METHOD | Subsample: $SUBSAMPLE"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TrainTime  | Trees/sec"
echo "--------|-------------|------------|------------|----------"

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) 开始计时
    start_ts=$(date +%s%3N)

    # 6.2) 执行 GBRT 并捕获完整 stdout
    output=$("$EXECUTABLE" "$DATA" \
        "$LOSS" $NUM_ITERATIONS $LEARNING_RATE $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$SPLIT_METHOD" $SUBSAMPLE 2>/dev/null)

    # 6.3) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 6.4) 提取关键指标
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

    # 计算吞吐量：树数/秒
    if [[ -n "$trees" && "$elapsed" -gt 0 ]]; then
        trees_per_sec=$(echo "scale=2; $trees * 1000 / $elapsed" | bc -l 2>/dev/null || echo "N/A")
    else
        trees_per_sec="N/A"
    fi

    # 处理空值
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$train_time" ]] && train_time="0"
    [[ -z "$trees" ]] && trees="0"

    # 6.5) 打印并记录本次结果行
    printf "%7d | %11d | %-10s | %-10s | %s\n" \
           "$t" "$elapsed" "$test_mse" "${train_time}ms" "$trees_per_sec"
done

echo ""
echo "==============================================="
echo "Strong Scaling Analysis:"
echo "- 理想: 线性加速，时间反比于线程数"
echo "- 关注点: TestMSE 保持稳定，Elapsed 时间下降"
echo "- 效率 = (串行时间 / 并行时间) / 线程数"
echo "- GBRT特点: 树构建具有数据依赖性，并行效果有限"
echo "==============================================="

exit 0