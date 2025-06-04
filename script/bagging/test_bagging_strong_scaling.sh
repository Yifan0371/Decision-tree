#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_strong_scaling.sh
#
# Bagging Strong-scaling test – 固定数据规模，按线程数测试耗时和 MSE
# =============================================================================

# 1) 项目根目录 & 可执行
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "错误：找不到可执行 $EXECUTABLE"
  exit 1
fi

# 2) 固定参数
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA" ]]; then
  echo "错误：找不到数据 $DATA"
  exit 1
fi

NUM_TREES=20
SAMPLE_RATIO=1.0
MAX_DEPTH=10
MIN_LEAF=2
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
PRUNER_PARAM=0.01
SEED=42

# 3) 生成线程列表
threads=(1)
while (( threads[-1]*2 <= OMP_NUM_THREADS )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != OMP_NUM_THREADS )) && threads+=( $OMP_NUM_THREADS )

# 4) 打印表头
echo "==============================================="
echo "    Bagging Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Trees: $NUM_TREES | Sample Ratio: $SAMPLE_RATIO"
echo "  Max Depth: $MAX_DEPTH | Min Leaf: $MIN_LEAF"
echo "  Criterion: $CRITERION | Finder: $FINDER | Pruner: $PRUNER"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TestMAE    | OOB_MSE    | Trees/sec"
echo "--------|-------------|------------|------------|------------|----------"

for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t
  start_ts=$(date +%s%3N)

  output=$("$EXECUTABLE" bagging "$DATA" \
      $NUM_TREES $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
      "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED)

  end_ts=$(date +%s%3N)
  elapsed=$(( end_ts - start_ts ))

  test_mse=$(echo "$output" | grep -E "Test MSE:" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  test_mae=$(echo "$output" | grep -E "Test MAE:" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
  oob_mse=$(echo "$output" | grep -E "OOB MSE:" | sed -n 's/.*OOB MSE: *\([0-9.-]*\).*/\1/p' | tail -1)

  [[ -z "$test_mse" ]] && test_mse="ERROR"
  [[ -z "$test_mae" ]] && test_mae="ERROR"
  [[ -z "$oob_mse" ]] && oob_mse="ERROR"

  if (( elapsed > 0 )); then
    trees_per_sec=$(echo "scale=2; $NUM_TREES * 1000 / $elapsed" | bc -l)
  else
    trees_per_sec="N/A"
  fi

  printf "%7d | %11d | %-10s | %-10s | %-10s | %s\n" \
         "$t" "$elapsed" "$test_mse" "$test_mae" "$oob_mse" "$trees_per_sec"
done

echo ""
echo "==============================================="
echo "Strong Scaling Analysis:"
echo "- 理想: 线程数×2→Elapsed 应减半，关注 TestMSE 保持稳定。"
echo "- 效率 = (单线程耗时 / 当前耗时) / 线程数。"
echo "==============================================="

exit 0
