#!/bin/bash
set -euo pipefail

# =============================================================================
# script/single_tree/test_singletree_strong_scaling.sh
#
# Strong-scaling test – 固定数据规模，按线程数测试耗时和 MSE
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

# 3) 生成线程列表：1, 2, 4, …, MAX_CORES；末尾保证有 MAX_CORES
MAX_CORES=$OMP_NUM_THREADS
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 4) 打印表头
echo "Threads | Elapsed(ms) | MSE"
echo "------------------------------------"

for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t
  start_ts=$(date +%s%3N)

  output=$("$EXECUTABLE" single "$DATA" \
      $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
      "$PRUNER" 0 "$VAL_SPLIT")

  end_ts=$(date +%s%3N)
  elapsed=$(( end_ts - start_ts ))

  # 提取“MSE: <num>”这一行
  mse=$(echo "$output" | grep -Eo "^MSE:\s*[0-9.+-eE]+" | awk -F'[: ]+' '{print $2}' | tail -1)
  [[ -z "$mse" ]] && mse="N/A"

  printf "%7d | %11d | %s\n" "$t" "$elapsed" "$mse"
done

exit 0
