#!/bin/bash
set -euo pipefail

# =============================================================================
# script/single_tree/test_singletree_weak_scaling.sh
#
# Weak-scaling test – 对同一份数据，截取不同大小子集，按线程数测试耗时和 MSE
# =============================================================================

# 1) 项目根路径 & 可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "❌ 错误：找不到可执行 $EXECUTABLE"
  exit 1
fi

# 2) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA" ]]; then
  echo "❌ 错误：找不到数据 $DATA"
  exit 1
fi

MAX_DEPTH=20
MIN_LEAF=5
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
VAL_SPLIT=0.2

# 3) 计算总行数（不含 header），基准行数 BASE
total_rows=$(( $(wc -l < "$DATA") - 1 ))
if (( total_rows < OMP_NUM_THREADS )); then
  echo "警告：数据行数 ($total_rows) 小于物理核数 ($OMP_NUM_THREADS)，脚本退出"
  exit 1
fi
BASE=$(( total_rows / OMP_NUM_THREADS ))

# 4) 生成线程列表：1,2,4,…,MAX_CORES；末尾加上 MAX_CORES
MAX_CORES=$OMP_NUM_THREADS
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 5) 打印表头
echo "Threads | SubsetRows | Elapsed(ms) | MSE"
echo "---------------------------------------------"

for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t

  # 计算本次截取的行数 chunk_size
  chunk_size=$(( t * BASE ))
  if (( chunk_size > total_rows )); then
    chunk_size=$total_rows
  fi
  lines_to_take=$(( chunk_size + 1 ))  # 包含 header

  tmpfile="$PROJECT_ROOT/data/data_clean/tmp_singletree_t${t}_$$.csv"
  head -n "$lines_to_take" "$DATA" > "$tmpfile"

  start_ts=$(date +%s%3N)
  output=$("$EXECUTABLE" single "$tmpfile" \
      $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
      "$PRUNER" 0 "$VAL_SPLIT")
  end_ts=$(date +%s%3N)
  elapsed=$(( end_ts - start_ts ))

  rm -f "$tmpfile"

  mse=$(echo "$output" | grep -Eo "^MSE:\s*[0-9.+-eE]+" | awk -F'[: ]+' '{print $2}' | tail -1)
  [[ -z "$mse" ]] && mse="N/A"

  printf "%7d | %10d | %11d | %s\n" "$t" "$chunk_size" "$elapsed" "$mse"
done

exit 0
