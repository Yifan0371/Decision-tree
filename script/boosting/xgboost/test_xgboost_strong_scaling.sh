#!/bin/bash
set -euo pipefail

# ==============================================================================
# script/boosting/xgboost/test_xgboost_strong_scaling.sh
# XGBoost Strong-scaling test – 固定数据规模，按线程数测试耗时和MSE
# ==============================================================================

# 1) 定位项目根目录 & 日志目录
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/xgboost"
mkdir -p "$SCRIPT_DIR"

# 定义并创建日志文件（用时间戳区分每次运行）
LOGFILE="$SCRIPT_DIR/xgboost_strong_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "XGBoost Strong Scaling Performance Test Log" >> "$LOGFILE"
echo "运行时间: $(date)" >> "$LOGFILE"
echo "项目根: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# 将后续所有输出同时写入终端和日志文件
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/XGBoostMain"

# 2) source env_config.sh 自动设置 OMP_NUM_THREADS，并在必要时编译
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
OBJECTIVE="reg:squarederror"
NUM_ROUNDS=30
ETA=0.3
MAX_DEPTH=6
MIN_CHILD_WEIGHT=1
LAMBDA=1.0
GAMMA=0.0
SUBSAMPLE=1.0
COLSAMPLE_BYTREE=1.0

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
echo "    XGBoost Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Objective: $OBJECTIVE | Rounds: $NUM_ROUNDS"
echo "  Eta: $ETA | Max Depth: $MAX_DEPTH"
echo "  Min Child Weight: $MIN_CHILD_WEIGHT | Lambda: $LAMBDA"
echo "  Gamma: $GAMMA | Subsample: $SUBSAMPLE"
echo "  ColSample: $COLSAMPLE_BYTREE"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TrainTime  | Trees/sec | Efficiency"
echo "--------|-------------|------------|------------|-----------|----------"

baseline_time=0

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) 开始计时
    start_ts=$(date +%s%3N)

    # 6.2) 执行 XGBoost 并捕获完整 stdout
    output=$("$EXECUTABLE" \
        --data "$DATA" \
        --objective "$OBJECTIVE" \
        --num-rounds $NUM_ROUNDS \
        --eta $ETA \
        --max-depth $MAX_DEPTH \
        --min-child-weight $MIN_CHILD_WEIGHT \
        --lambda $LAMBDA \
        --gamma $GAMMA \
        --subsample $SUBSAMPLE \
        --colsample-bytree $COLSAMPLE_BYTREE \
        --quiet 2>/dev/null)

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

    # 计算效率
    if [ $t -eq 1 ]; then
        baseline_time=$elapsed
        efficiency="1.00"
    else
        if [ $elapsed -gt 0 ] && [ $baseline_time -gt 0 ]; then
            speedup=$(echo "scale=2; $baseline_time / $elapsed" | bc -l 2>/dev/null || echo "N/A")
            efficiency=$(echo "scale=2; $speedup / $t" | bc -l 2>/dev/null || echo "N/A")
        else
            efficiency="N/A"
        fi
    fi

    # 处理空值
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$train_time" ]] && train_time="0"
    [[ -z "$trees" ]] && trees="0"

    # 6.5) 打印并记录本次结果行
    printf "%7d | %11d | %-10s | %-10s | %-9s | %s\n" \
           "$t" "$elapsed" "$test_mse" "${train_time}ms" "$trees_per_sec" "$efficiency"
done

echo ""
echo "==============================================="
echo "XGBoost Strong Scaling Analysis:"
echo "- 理想: 线性加速，时间反比于线程数"
echo "- 关注点: TestMSE 保持稳定，Elapsed 时间下降"
echo "- 效率 = (串行时间 / 并行时间) / 线程数"
echo ""
echo "XGBoost并行特性:"
echo "- 预排序并行: 高效率，接近线性扩展"
echo "- 梯度计算并行: 完美并行，效率接近1.0"
echo "- 树构建并行: 中等并行度，受分裂查找限制"
echo "- 整体预期效率: 0.8-0.95（优于传统GBRT）"
echo "==============================================="

exit 0