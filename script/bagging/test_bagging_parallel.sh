#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_parallel.sh
# Bagging 并行性能全面测试脚本
# =============================================================================

# 1) 项目根路径 & 设置线程数
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: 找不到可执行 $EXECUTABLE，请先编译"
  exit 1
fi

# 2) 数据路径 & 结果文件
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: 数据 $DATA_PATH 不存在"
  exit 1
fi

RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/parallel_performance_results.txt"
> "$RESULTS_FILE"

echo "=========================================="
echo "    Bagging Parallel Performance Test     "
echo "=========================================="
echo "物理核数: $OMP_NUM_THREADS"
echo "数据文件: $(basename "$DATA_PATH")"
echo "结果保存: $RESULTS_FILE"
echo "时间: $(date)"
echo ""

# 3) 生成线程列表
threads_list=(1)
cur=1
while (( cur*2 <= OMP_NUM_THREADS )); do
  cur=$((cur*2))
  threads_list+=( $cur )
done
if (( cur != OMP_NUM_THREADS )); then
  threads_list+=( $OMP_NUM_THREADS )
fi
echo "测试线程数序列: ${threads_list[*]}"
echo ""

# 4) 结果文件表头
{
  echo "# Bagging Parallel Performance Test Results"
  echo "# Date: $(date)"
  echo "# Max Cores: $OMP_NUM_THREADS"
  echo "# Data: $(basename "$DATA_PATH")"
  echo "# Format: Config,Threads,TestMSE,TestMAE,OOB_MSE,TrainTime(ms),TotalTime(ms),Speedup,Efficiency"
} >> "$RESULTS_FILE"

declare -A baseline_times

extract_results() {
  local out="$1"
  local test_mse test_mae oob_mse train_time total_time
  test_mse=$(echo "$out" | grep -E "Test MSE:" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  test_mae=$(echo "$out" | grep -E "Test MAE:" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
  oob_mse=$(echo "$out" | grep -E "OOB MSE:" | sed -n 's/.*OOB MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  train_time=$(echo "$out" | grep -E "Train Time:" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
  total_time=$(echo "$out" | grep -E "Total Time:" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)

  [[ -z "$test_mse" ]] && test_mse="ERROR"
  [[ -z "$test_mae" ]] && test_mae="ERROR"
  [[ -z "$oob_mse" ]] && oob_mse="ERROR"
  [[ -z "$train_time" ]] && train_time="0"
  [[ -z "$total_time" ]] && total_time="0"

  echo "$test_mse,$test_mae,$oob_mse,$train_time,$total_time"
}

run_test() {
  local config_name="$1"
  local threads="$2"
  shift 2
  local params=( "$@" )

  export OMP_NUM_THREADS=$threads
  echo -n "  测试 $config_name ($threads 线程)... "

  local start_ts=$(date +%s%3N)
  local out
  out=$("$EXECUTABLE" bagging "$DATA_PATH" "${params[@]}" 2>&1)
  local exit_code=$?
  local end_ts=$(date +%s%3N)
  local wall_time=$(( end_ts - start_ts ))

  if (( exit_code != 0 )); then
    echo "FAILED"
    echo "$config_name,$threads,ERROR,ERROR,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
    return
  fi

  local results
  results=$(extract_results "$out")
  local test_mse test_mae oob_mse train_time total_time
  IFS=',' read -r test_mse test_mae oob_mse train_time total_time <<< "$results"
  echo "完成 (TrainTime=${train_time}ms)"

  local speedup="N/A"
  local efficiency="N/A"
  if (( threads == 1 )); then
    baseline_times["$config_name"]="$train_time"
    speedup="1.00"
    efficiency="1.00"
  else
    local base="${baseline_times[$config_name]:-0}"
    if (( base > 0 && train_time > 0 )); then
      speedup=$(echo "scale=2; $base / $train_time" | bc -l)
      efficiency=$(echo "scale=2; $speedup / $threads" | bc -l)
    fi
  fi

  echo "$config_name,$threads,$test_mse,$test_mae,$oob_mse,$train_time,$wall_time,$speedup,$efficiency" \
    >> "$RESULTS_FILE"
}

# 5) 定义各配置
declare -A test_configs
test_configs["FastSmall"]="10 1.0 8 2 mse exhaustive none 0.01 42"
test_configs["Standard"]="20 1.0 10 2 mse exhaustive none 0.01 42"
test_configs["DeepTrees"]="15 1.0 15 1 mse exhaustive none 0.01 42"
test_configs["RandomSplit"]="20 1.0 10 2 mse random none 0.01 42"
test_configs["HistogramSplit"]="20 1.0 10 2 mse histogram_ew:64 none 0.01 42"

echo "开始性能测试..."
echo ""

for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit"; do
  echo "=== 配置: $config_name ==="
  IFS=' ' read -r -a params <<< "${test_configs[$config_name]}"
  for threads in "${threads_list[@]}"; do
    run_test "$config_name" "$threads" "${params[@]}"
  done
  echo ""
done

echo "=========================================="
echo "测试完成！结果已保存到: $RESULTS_FILE"
echo "=========================================="
