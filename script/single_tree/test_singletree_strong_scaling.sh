#!/bin/bash
set -euo pipefail

# ===============================================================
# Strong-scaling test – 固定数据规模，按线程数测试耗时和 MSE
# ===============================================================

# 1) 项目根路径 & 可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"

# 2) source env_config.sh 自动设置 OMP_NUM_THREADS，并在必要时编译
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
MAX_DEPTH=20
MIN_LEAF=5
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
VAL_SPLIT=0.2   # 验证集比例

# 4) 确认可执行和数据文件存在
[[ -f "$EXECUTABLE" ]] || { echo "❌ $EXECUTABLE 不存在"; exit 1; }
[[ -f "$DATA"       ]] || { echo "❌ $DATA 不存在"; exit 1; }

# 5) 生成线程列表：1, 2, 4, …, MAX_CORES，如果最后一个不是 MAX_CORES 再加一个
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
    threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 6) 打印表头：Threads | Elapsed(ms) | MSE
echo "Threads | Elapsed(ms) | MSE"
echo "------------------------------------"

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) 开始计时
    start_ts=$(date +%s%3N)

    # 6.2) 执行可执行并捕获完整 stdout
    output=$("$EXECUTABLE" single "$DATA" \
        $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
        "$PRUNER" 0 "$VAL_SPLIT")

    # 6.3) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 6.4) 从 output 中提取 MSE（只匹配以“MSE:”开头的那一行）
    mse=$(echo "$output" | grep -i "^MSE:" | awk -F'[: ]+' '{print $2}')

    # 6.5) 打印结果行
    printf "%7d | %11d | %s\n" "$t" "$elapsed" "$mse"
done

exit 0
