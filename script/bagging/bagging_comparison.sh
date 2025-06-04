#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_performance_comparison.sh
#
# 比较 Exhaustive vs Random Split Finders 在 Bagging 模块下的性能差异
# =============================================================================

# 1) 项目根路径 & 设置线程数
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: 找不到可执行 $EXECUTABLE，请先编译"
  exit 1
fi

# 2) 数据路径 & 输出文件
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: 找不到数据文件 $DATA_PATH"
  exit 1
fi

RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

EXHAUSTIVE_RESULTS="$RESULTS_DIR/exhaustive_results.txt"
RANDOM_RESULTS="$RESULTS_DIR/random_results.txt"

# 清空旧结果并写入表头
{
  echo "# Exhaustive Finder Results - $(date)"
  echo "Trees,Train_Time(ms),Test_MSE,Test_MAE,OOB_MSE"
} > "$EXHAUSTIVE_RESULTS"

{
  echo "# Random Finder Results - $(date)"
  echo "Trees,Train_Time(ms),Test_MSE,Test_MAE,OOB_MSE"
} > "$RANDOM_RESULTS"

TREE_COUNTS=(5 10 20 50 100)

echo "=========================================="
echo "Bagging Performance Comparison"
echo "=========================================="
echo "Comparing: Exhaustive vs Random Finders"
echo "Configuration: MSE criterion, No pruner"
echo "Date: $(date)"
echo ""
echo "Testing configurations: Trees = ${TREE_COUNTS[*]}"
echo "Data: $DATA_PATH"
echo "Timeout: 300 seconds per test"
echo ""

# 提取结果的辅助函数
extract_results() {
  local log_file="$1"
  local train_time test_mse test_mae oob_mse
  train_time=$(grep -E "Train Time:" "$log_file" | awk '{print $3}' | sed 's/ms//g')
  test_mse=$(grep -E "Test MSE:" "$log_file" | awk '{print $3}')
  test_mae=$(grep -E "Test MAE:" "$log_file" | awk '{print $6}')
  oob_mse=$(grep -E "OOB MSE:" "$log_file" | awk '{print $3}')
  echo "$train_time,$test_mse,$test_mae,$oob_mse"
}

for trees in "${TREE_COUNTS[@]}"; do
  echo "Testing with $trees trees..."

  # Exhaustive
  echo "  Running Exhaustive finder..."
  temp_ex="temp_exhaustive_${trees}.log"
  timeout 300 "$EXECUTABLE" bagging "$DATA_PATH" \
    "$trees" 1.0 30 2 mse exhaustive none 0.01 42 \
    > "$temp_ex" 2>&1

  if [[ $? -eq 0 ]]; then
    results=$(extract_results "$temp_ex")
    echo "$trees,$results" >> "$EXHAUSTIVE_RESULTS"
    echo "    Exhaustive completed: $results"
  else
    echo "    Exhaustive failed或超时"
    echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$EXHAUSTIVE_RESULTS"
  fi
  rm -f "$temp_ex"

  # Random
  echo "  Running Random finder..."
  temp_rnd="temp_random_${trees}.log"
  timeout 300 "$EXECUTABLE" bagging "$DATA_PATH" \
    "$trees" 1.0 30 2 mse random none 0.01 42 \
    > "$temp_rnd" 2>&1

  if [[ $? -eq 0 ]]; then
    results=$(extract_results "$temp_rnd")
    echo "$trees,$results" >> "$RANDOM_RESULTS"
    echo "    Random completed: $results"
  else
    echo "    Random failed或超时"
    echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$RANDOM_RESULTS"
  fi
  rm -f "$temp_rnd"

  echo "  Completed $trees trees test"
  echo ""
done

echo "=========================================="
echo "Summary Report"
echo "=========================================="
echo ""
echo "Exhaustive Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
tail -n +3 "$EXHAUSTIVE_RESULTS" | while IFS=',' read -r t tt mse mae oob; do
  printf "%5s | %10s | %8s | %8s | %7s\n" "$t" "$tt" "$mse" "$mae" "$oob"
done

echo ""
echo "Random Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
tail -n +3 "$RANDOM_RESULTS" | while IFS=',' read -r t tt mse mae oob; do
  printf "%5s | %10s | %8s | %8s | %7s\n" "$t" "$tt" "$mse" "$mae" "$oob"
done

echo ""
echo "Results saved to:"
echo "  - $EXHAUSTIVE_RESULTS"
echo "  - $RANDOM_RESULTS"
echo ""
echo "Performance comparison completed."
echo "=========================================="
