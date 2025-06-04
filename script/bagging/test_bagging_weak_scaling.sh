#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_weak_scaling.sh
#
# Bagging Weak-scaling test – 对同一份数据，截取不同大小子集，按线程数测试耗时和 MSE
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

# 3) 总行数 & 基准行 BASE
total_rows=$(( $(wc -l < "$DATA") - 1 ))
if (( total_rows < OMP_NUM_THREADS )); then
  echo "警告：数据行数 ($total_rows) 小于物理核数 ($OMP_NUM_THREADS)，脚本退出"
  exit 1
fi
BASE=$(( total_rows / OMP_NUM_THREADS ))
echo "总行数 (不含 header): $total_rows，物理核数: $OMP_NUM_THREADS，基准行数 BASE=$BASE"

# 4) 生成线程列表：1,2,4,…,MAX_CORES；末尾加上 MAX_CORES
threads=(1)
while (( threads[-1]*2 <= OMP_NUM_THREADS )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != OMP_NUM_THREADS )) && threads+=( $OMP_NUM_THREADS )

# 5) 打印表头
echo "==============================================="
echo "    Bagging Weak Scaling Performance Test     "
echo "==============================================="
echo "Fixed Parameters (per thread):"
echo "  Trees: $NUM_TREES | Sample Ratio: $SAMPLE_RATIO"
echo "  Max Depth: $MAX_DEPTH | Min Leaf: $MIN_LEAF"
echo "  Criterion: $CRITERION | Finder: $FINDER"
echo "  Base rows per thread: $BASE"
echo ""
echo "Threads | SubsetRows | Elapsed(ms) | TestMSE    | TestMAE    | OOB_MSE    | Efficiency"
echo "--------|------------|-------------|------------|------------|------------|----------"

baseline_time=0
for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t

  chunk_size=$(( t * BASE ))
  if (( chunk_size > total_rows )); then
    chunk_size=$total_rows
  fi
  lines_to_take=$(( chunk_size + 1 ))

  tmpfile="$PROJECT_ROOT/data/data_clean/tmp_bagging_t${t}_$$.csv"
  head -n "$lines_to_take" "$DATA" > "$tmpfile"

  start_ts=$(date +%s%3N)
  output=$("$EXECUTABLE" bagging "$tmpfile" \
      $NUM_TREES $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
      "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED)
  end_ts=$(date +%s%3N)
  elapsed=$(( end_ts - start_ts ))

  rm -f "$tmpfile"

  test_mse=$(echo "$output" | grep -E "Test MSE:" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  test_mae=$(echo "$output" | grep -E "Test MAE:" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
  oob_mse=$(echo "$output" | grep -E "OOB MSE:" | sed -n 's/.*OOB MSE: *\([0-9.-]*\).*/\1/p' | tail -1)

  [[ -z "$test_mse" ]] && test_mse="ERROR"
  [[ -z "$test_mae" ]] && test_mae="ERROR"
  [[ -z "$oob_mse" ]] && oob_mse="ERROR"

  if (( t == 1 )); then
    baseline_time=$elapsed
    efficiency="1.00"
  else
    if (( baseline_time > 0 && elapsed > 0 )); then
      efficiency=$(echo "scale=2; $baseline_time / $elapsed" | bc -l)
    else
      efficiency="N/A"
    fi
  fi

  printf "%7d | %10d | %11d | %-10s | %-10s | %-10s | %s\n" \
         "$t" "$chunk_size" "$elapsed" "$test_mse" "$test_mae" "$oob_mse" "$efficiency"
done

echo ""
echo "==============================================="
echo "Weak Scaling Analysis:"
echo "- 理想: 线程数增加，处理更多数据但时间恒定"
echo "- 效率 = (单线程时间 / 当前时间)"
echo "- 关注: 效率 接近 1.0 表示良好"
echo "==============================================="

exit 0
