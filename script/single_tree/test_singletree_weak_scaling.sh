#!/bin/bash
set -euo pipefail

# ======================================================================
# Weak-scaling test（自动按行截取同一大文件）– 对同一份数据，截取不同大小子集，按线程数测试耗时和 MSE
# ======================================================================

# 1) 定位项目根目录和可执行
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"

# 2) source env_config.sh：自动设置 OMP_NUM_THREADS，并在必要时编译可执行
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) 固定使用的原始数据文件：cleaned_data.csv
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
MAX_DEPTH=20
MIN_LEAF=5
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
VAL_SPLIT=0.2   # 验证集比例，可根据需求改为 0 或 0.2

# 4) 确认可执行和数据都存在
[[ -f "$EXECUTABLE" ]] || { echo "❌ $EXECUTABLE 不存在"; exit 1; }
[[ -f "$DATA"       ]] || { echo "❌ $DATA 不存在"; exit 1; }

# 5) 计算原始数据的总行数（不含 header）和基准行数 BASE
#    wc -l 会算上 header，所以减 1
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

# 7) 打印结果表头：Threads | SubsetRows | Elapsed(ms) | MSE
echo "Threads | SubsetRows | Elapsed(ms) | MSE"
echo "-------------------------------------------------"

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
    tmpfile="$PROJECT_ROOT/data/data_clean/tmp_chunk_t${t}.csv"
    # 保证前一次运行的 tmpfile 被覆盖
    head -n "$lines_to_take" "$DATA" > "$tmpfile"

    # 8.3) 开始计时
    start_ts=$(date +%s%3N)

    # 8.4) 运行 DecisionTreeMain，并捕获完整 stdout
    output=$("$EXECUTABLE" single "$tmpfile" \
        $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
        "$PRUNER" 0 "$VAL_SPLIT")

    # 8.5) 结束计时
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 8.6) 提取 MSE（只匹配以“MSE:”开头的行）
    mse=$(echo "$output" | grep -i "^MSE:" | awk -F'[: ]+' '{print $2}')

    # 8.7) 打印本次结果行：线程数 | 本次截取行数 | 耗时(ms) | MSE
    printf "%7d | %10d | %11d | %s\n" \
           "$t" "$chunk_size" "$elapsed" "$mse"

    # 8.8) 删除临时文件（可选，也可以留作后续检查）
    rm -f "$tmpfile"
done

exit 0
