#!/bin/bash
set -euo pipefail

# ==============================================================================
# Bagging Strong-scaling test – 固定数据规模，按线程数测试耗时和 MSE，
# 并将结果保存到日志文件
# ==============================================================================

# 1) 定位项目根目录 & 日志目录
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$SCRIPT_DIR"

# 定义并创建日志文件（用时间戳区分每次运行）
LOGFILE="$SCRIPT_DIR/bagging_strong_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "Bagging Strong Scaling Performance Test Log" >> "$LOGFILE"
echo "运行时间: $(date)" >> "$LOGFILE"
echo "项目根: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# 将后续所有输出同时写入终端和日志文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"

# 2) source env_config.sh 自动设置 OMP_NUM_THREADS，并在必要时编译
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
NUM_TREES=20
SAMPLE_RATIO=1.0
MAX_DEPTH=10
MIN_LEAF=2
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
PRUNER_PARAM=0.01
SEED=42

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
echo "    Bagging Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Trees: $NUM_TREES | Sample Ratio: $SAMPLE_RATIO"
echo "  Max Depth: $MAX_DEPTH | Min Leaf: $MIN_LEAF"
echo "  Criterion: $CRITERION | Finder: $FINDER"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TestMAE    | OOB_MSE    | Trees/sec"
echo "--------|-------------|------------|------------|------------|----------"

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) 开始计时
    start_ts=$(date +%s%3N)

    # 6.2) 执行 bagging 并捕获完整 stdout
    output=$("$EXECUTABLE" bagging "$DATA" \
        $NUM_TREES $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED 2>/dev/null)

    # 6.3) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 6.4) 提取关键指标
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(echo "$output" | grep "Test MAE:" | sed -n 's/.*Test MAE: \([0-9.-]*\).*/\1/p' | tail -1)
    oob_mse=$(echo "$output" | grep "OOB MSE:" | sed -n 's/.*OOB MSE: \([0-9.-]*\).*/\1/p' | tail -1)

    # 计算吞吐量：树数/秒
    trees_per_sec=$(echo "scale=2; $NUM_TREES * 1000 / $elapsed" | bc -l 2>/dev/null || echo "N/A")

    # 处理空值
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    [[ -z "$oob_mse" ]] && oob_mse="ERROR"

    # 6.5) 打印并记录本次结果行
    printf "%7d | %11d | %-10s | %-10s | %-10s | %s\n" \
           "$t" "$elapsed" "$test_mse" "$test_mae" "$oob_mse" "$trees_per_sec"
done

echo ""
echo "==============================================="
echo "Strong Scaling Analysis:"
echo "- 理想: 线性加速，时间反比于线程数"
echo "- 关注点: TestMSE 保持稳定，Elapsed 时间下降"
echo "- 效率 = (串行时间 / 并行时间) / 线程数"
echo "==============================================="

exit 0
