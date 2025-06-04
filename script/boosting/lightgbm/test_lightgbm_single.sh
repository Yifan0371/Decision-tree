#!/bin/bash
# =============================================================================
# script/boosting/lightgbm/test_lightgbm_single.sh
#
# 单线程模式下，对 LightGBM 模块做全量测试（各种参数配置网格）
# 只输出每次运行的"配置 | TestMSE | TestMAE | Train(ms) | Total(ms)"，并汇总到：
#   script/boosting/lightgbm/lightgbm_single_results_TIMESTAMP.txt
#
# 使用：在项目根目录运行
#   bash script/boosting/lightgbm/test_lightgbm_single.sh
# =============================================================================

export OMP_NUM_THREADS=1

# 项目根路径和可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# source env_config.sh 确保项目已构建
source "$PROJECT_ROOT/script/env_config.sh"

DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/LightGBMMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/lightgbm"
mkdir -p "$RESULTS_DIR"

OUTFILE="$RESULTS_DIR/lightgbm_single_results_$(date +%Y%m%d_%H%M%S).txt"
echo "配置 | TestMSE | TestMAE | Train(ms) | Total(ms)" > "$OUTFILE"

if [ ! -f "$EXECUTABLE" ]; then
  echo "错误：找不到可执行文件 $EXECUTABLE"
  exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：找不到数据文件 $DATA_PATH"
  # 尝试其他数据文件
  for alt_data in "$PROJECT_ROOT/data/data_clean/cleaned_sample_400_rows.csv" "$PROJECT_ROOT/data/data_clean/cleaned_15k_random.csv"; do
      if [ -f "$alt_data" ]; then
          echo "使用替代数据文件: $(basename $alt_data)"
          DATA_PATH="$alt_data"
          break
      fi
  done
  
  if [ ! -f "$DATA_PATH" ]; then
      echo "未找到任何可用的数据文件"
      exit 1
  fi
fi

echo "=========================================="
echo "    LightGBM Single Thread Test Suite     "
echo "=========================================="
echo "数据文件: $(basename $DATA_PATH)"
echo "结果保存: $OUTFILE"
echo "时间: $(date)"
echo ""

# --- 帮助函数：运行并解析一行 Summary ---
run_and_parse() {
  local config_desc="$1"       # 比如 "SplitMethod=histogram_eq:64,Iterations=100"
  shift
  local args=( "$@" )          # 传给 LightGBMMain 的参数

  # 执行程序并捕获输出
  local start_time=$(date +%s%3N)
  local raw
  raw=$("$EXECUTABLE" "${args[@]}" 2>&1)
  local end_time=$(date +%s%3N)
  local wall_time=$((end_time - start_time))

  # 从 raw 中提取关键字段
  local test_mse=$(echo "$raw" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
  local test_mae=$(echo "$raw" | grep "Test MAE:" | sed -n 's/.*Test MAE: \([0-9.-]*\).*/\1/p' | tail -1)
  local train_time=$(echo "$raw" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)

  if [[ -z "$test_mse" || -z "$test_mae" || -z "$train_time" ]]; then
    # 如果没有找到，写一个 N/A 占位
    echo "${config_desc} | N/A | N/A | N/A | ${wall_time}" >> "$OUTFILE"
  else
    echo "${config_desc} | ${test_mse} | ${test_mae} | ${train_time} | ${wall_time}" >> "$OUTFILE"
  fi
}

###############################################################################
# 1. Split Method 测试
###############################################################################
echo "=== 1. Split Method 测试 ==="
SPLIT_METHODS=("histogram_eq:32" "histogram_eq:64" "histogram_eq:128" "histogram_ew:32" "histogram_ew:64" "histogram_ew:128" "adaptive_ew:sturges" "adaptive_ew:rice" "adaptive_eq")

for method in "${SPLIT_METHODS[@]}"; do
  config="SplitMethod=${method}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "$method" \
    --num-iterations 100 \
    --learning-rate 0.1 \
    --num-leaves 31 \
    --min-data-in-leaf 20 \
    --quiet
done

###############################################################################
# 2. 迭代次数测试
###############################################################################
echo ""
echo "=== 2. 迭代次数测试 ==="
ITERATIONS=(50 100 150 200 300)

for iters in "${ITERATIONS[@]}"; do
  config="Iterations=${iters}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations "$iters" \
    --learning-rate 0.1 \
    --num-leaves 31 \
    --min-data-in-leaf 20 \
    --quiet
done

###############################################################################
# 3. 学习率测试
###############################################################################
echo ""
echo "=== 3. 学习率测试 ==="
LEARNING_RATES=(0.01 0.05 0.1 0.2 0.3 0.5)

for lr in "${LEARNING_RATES[@]}"; do
  config="LearningRate=${lr}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations 100 \
    --learning-rate "$lr" \
    --num-leaves 31 \
    --min-data-in-leaf 20 \
    --quiet
done

###############################################################################
# 4. 叶子数测试
###############################################################################
echo ""
echo "=== 4. 叶子数测试 ==="
NUM_LEAVES=(15 31 63 127 255)

for leaves in "${NUM_LEAVES[@]}"; do
  config="NumLeaves=${leaves}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations 100 \
    --learning-rate 0.1 \
    --num-leaves "$leaves" \
    --min-data-in-leaf 20 \
    --quiet
done

###############################################################################
# 5. 最小数据量测试
###############################################################################
echo ""
echo "=== 5. 最小数据量测试 ==="
MIN_DATA_VALUES=(5 10 20 50 100)

for min_data in "${MIN_DATA_VALUES[@]}"; do
  config="MinDataInLeaf=${min_data}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations 100 \
    --learning-rate 0.1 \
    --num-leaves 31 \
    --min-data-in-leaf "$min_data" \
    --quiet
done

###############################################################################
# 6. GOSS参数测试
###############################################################################
echo ""
echo "=== 6. GOSS参数测试 ==="
# 基准：无GOSS
config="GOSS=disabled"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "histogram_eq:64" \
  --num-iterations 100 \
  --learning-rate 0.1 \
  --num-leaves 31 \
  --min-data-in-leaf 20 \
  --disable-goss \
  --quiet

# GOSS参数组合
GOSS_CONFIGS=(
  "0.2,0.1"
  "0.3,0.1" 
  "0.2,0.2"
  "0.1,0.1"
  "0.4,0.1"
)

for goss_config in "${GOSS_CONFIGS[@]}"; do
  IFS=',' read -r top_rate other_rate <<< "$goss_config"
  config="GOSS=enabled,TopRate=${top_rate},OtherRate=${other_rate}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations 100 \
    --learning-rate 0.1 \
    --num-leaves 31 \
    --min-data-in-leaf 20 \
    --top-rate "$top_rate" \
    --other-rate "$other_rate" \
    --enable-goss \
    --quiet
done

###############################################################################
# 7. EFB(特征绑定)测试
###############################################################################
echo ""
echo "=== 7. EFB(特征绑定)测试 ==="
# EFB禁用
config="EFB=disabled"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "histogram_eq:64" \
  --num-iterations 100 \
  --learning-rate 0.1 \
  --num-leaves 31 \
  --min-data-in-leaf 20 \
  --disable-bundling \
  --quiet

# EFB启用（不同冲突率）
CONFLICT_RATES=(0.0 0.1 0.2 0.3)

for conflict_rate in "${CONFLICT_RATES[@]}"; do
  config="EFB=enabled,ConflictRate=${conflict_rate}"
  echo "测试: $config"
  run_and_parse "$config" \
    --data "$DATA_PATH" \
    --split-method "histogram_eq:64" \
    --num-iterations 100 \
    --learning-rate 0.1 \
    --num-leaves 31 \
    --min-data-in-leaf 20 \
    --max-conflict "$conflict_rate" \
    --enable-bundling \
    --quiet
done

###############################################################################
# 8. 组合优化测试
###############################################################################
echo ""
echo "=== 8. 组合优化测试 ==="

# 高精度配置
config="HighAccuracy"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "histogram_eq:128" \
  --num-iterations 200 \
  --learning-rate 0.05 \
  --num-leaves 63 \
  --min-data-in-leaf 10 \
  --top-rate 0.1 \
  --other-rate 0.05 \
  --enable-goss \
  --enable-bundling \
  --quiet

# 快速训练配置
config="FastTraining"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "histogram_ew:32" \
  --num-iterations 50 \
  --learning-rate 0.3 \
  --num-leaves 15 \
  --min-data-in-leaf 50 \
  --top-rate 0.3 \
  --other-rate 0.2 \
  --enable-goss \
  --enable-bundling \
  --quiet

# 平衡配置
config="Balanced"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "adaptive_ew:sturges" \
  --num-iterations 100 \
  --learning-rate 0.1 \
  --num-leaves 31 \
  --min-data-in-leaf 20 \
  --top-rate 0.2 \
  --other-rate 0.1 \
  --enable-goss \
  --enable-bundling \
  --quiet

# 最优配置（基于已知好的参数）
config="Optimized"
echo "测试: $config"
run_and_parse "$config" \
  --data "$DATA_PATH" \
  --split-method "histogram_eq:64" \
  --num-iterations 150 \
  --learning-rate 0.1 \
  --num-leaves 31 \
  --min-data-in-leaf 15 \
  --top-rate 0.2 \
  --other-rate 0.1 \
  --max-conflict 0.1 \
  --enable-goss \
  --enable-bundling \
  --quiet

echo ""
echo "=========================================="
echo "LightGBM单线程全面测试完成"
echo "=========================================="
echo "结果保存在: $OUTFILE"
echo ""

# 显示最佳配置
echo "=== 性能分析 ==="
echo "正在分析最佳配置..."

# 分析MSE最低的配置
echo ""
echo "MSE最低的前5个配置:"
echo "排名 | 配置 | TestMSE | TestMAE | Train(ms)"
echo "----|------|---------|---------|----------"

# 跳过header，按MSE排序
tail -n +2 "$OUTFILE" | grep -v "N/A" | sort -t'|' -k2 -n | head -5 | nl -w3 | while read num line; do
    config=$(echo "$line" | cut -d'|' -f1 | xargs)
    mse=$(echo "$line" | cut -d'|' -f2 | xargs)
    mae=$(echo "$line" | cut -d'|' -f3 | xargs)
    train_time=$(echo "$line" | cut -d'|' -f4 | xargs)
    printf "%3s | %-30s | %-7s | %-7s | %s\n" "$num" "$config" "$mse" "$mae" "$train_time"
done

echo ""
echo "训练时间最短的前5个配置:"
echo "排名 | 配置 | TestMSE | TestMAE | Train(ms)"
echo "----|------|---------|---------|----------"

# 按训练时间排序
tail -n +2 "$OUTFILE" | grep -v "N/A" | sort -t'|' -k4 -n | head -5 | nl -w3 | while read num line; do
    config=$(echo "$line" | cut -d'|' -f1 | xargs)
    mse=$(echo "$line" | cut -d'|' -f2 | xargs)
    mae=$(echo "$line" | cut -d'|' -f3 | xargs)
    train_time=$(echo "$line" | cut -d'|' -f4 | xargs)
    printf "%3s | %-30s | %-7s | %-7s | %s\n" "$num" "$config" "$mse" "$mae" "$train_time"
done

echo ""
echo "建议:"
echo "- 查看MSE最低的配置用于追求最佳精度"
echo "- 查看训练时间最短的配置用于快速原型开发"
echo "- 在精度和速度之间选择平衡的配置"
echo "- GOSS和EFB通常能在保持精度的同时提升速度"
echo ""
echo "详细结果请查看: $OUTFILE"