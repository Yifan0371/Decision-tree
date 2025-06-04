#!/bin/bash
# =============================================================================
# script/boosting/gbrt_dart/test_gbrt_dart_single.sh
# GBRT DART (带DART) 单线程全面测试脚本
# 在项目根目录运行: bash script/boosting/gbrt_dart/test_gbrt_dart_single.sh
# =============================================================================

export OMP_NUM_THREADS=1

PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt_dart"
OUTFILE="$RESULTS_DIR/gbrt_dart_single_results.txt"
mkdir -p "$RESULTS_DIR"

echo "配置 | TestMSE | TrainTime(ms) | TotalTime(ms) | Trees" > "$OUTFILE"

if [ ! -f "$EXECUTABLE" ]; then
  echo "错误：找不到可执行文件 $EXECUTABLE"
  exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：找不到数据文件 $DATA_PATH"
  exit 1
fi

# 运行并解析结果的函数
run_and_parse() {
  local config_desc="$1"
  shift
  local args=( "$@" )

  local start_time=$(date +%s%3N)
  local raw
  raw=$("$EXECUTABLE" "${args[@]}" 2>&1)
  local end_time=$(date +%s%3N)
  local wall_time=$((end_time - start_time))

  # 从输出中提取关键指标
  local test_mse=$(echo "$raw" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
  local train_time=$(echo "$raw" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
  local trees=$(echo "$raw" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

  # 处理空值
  [ -z "$test_mse" ] && test_mse="N/A"
  [ -z "$train_time" ] && train_time="N/A"
  [ -z "$trees" ] && trees="N/A"

  echo "${config_desc} | ${test_mse} | ${train_time} | ${wall_time} | ${trees}" >> "$OUTFILE"
}

echo "=== GBRT DART 单线程全面测试 ==="
echo "数据文件: $DATA_PATH"
echo "结果保存: $OUTFILE"
echo ""

###############################################################################
# 1. DART vs 标准GBRT基准对比
###############################################################################
echo "1. DART vs 标准GBRT对比..."
run_and_parse "Standard GBRT,基准" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "false" 0.0 "false" "false"
run_and_parse "DART 0%dropout,一致性" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.0 "false" "false"

###############################################################################
# 2. DART丢弃率敏感性测试
###############################################################################
echo "2. DART丢弃率测试..."
run_and_parse "DART 5%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.05 "false" "false"
run_and_parse "DART 10%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.10 "false" "false"
run_and_parse "DART 15%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.15 "false" "false"
run_and_parse "DART 20%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART 25%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.25 "false" "false"
run_and_parse "DART 30%dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.30 "false" "false"

###############################################################################
# 3. DART权重归一化测试
###############################################################################
echo "3. DART权重归一化测试..."
run_and_parse "DART norm=true,归一化开启" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "true" "false"
run_and_parse "DART norm=false,归一化关闭" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 4. DART预测模式测试
###############################################################################
echo "4. DART预测模式测试..."
run_and_parse "DART skip=false,训练模式" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART skip=true,预测模式" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "true"

###############################################################################
# 5. DART与不同损失函数组合
###############################################################################
echo "5. DART与损失函数组合..."
run_and_parse "DART+Squared Loss" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Huber Loss" "$DATA_PATH" "huber" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Absolute Loss" "$DATA_PATH" "absolute" 30 0.1 4 1 "mae" "exhaustive" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 6. DART与不同树深度组合
###############################################################################
echo "6. DART与树深度组合..."
run_and_parse "DART depth=2,浅树" "$DATA_PATH" "squared" 30 0.1 2 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=3,平衡树" "$DATA_PATH" "squared" 30 0.1 3 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=4,标准树" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=6,深树" "$DATA_PATH" "squared" 30 0.1 6 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 7. DART与不同学习率组合
###############################################################################
echo "7. DART与学习率组合..."
run_and_parse "DART lr=0.05,保守学习" "$DATA_PATH" "squared" 40 0.05 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.1,标准学习" "$DATA_PATH" "squared" 40 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.2,快速学习" "$DATA_PATH" "squared" 40 0.2 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.3,激进学习" "$DATA_PATH" "squared" 40 0.3 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 8. DART与分割方法组合
###############################################################################
echo "8. DART与分割方法组合..."
run_and_parse "DART+穷举分割" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+随机分割" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "random" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+等宽直方图" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_ew" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 9. DART与子采样组合
###############################################################################
echo "9. DART与子采样组合..."
run_and_parse "DART+50%子采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.5 "true" 0.20 "false" "false"
run_and_parse "DART+70%子采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.7 "true" 0.20 "false" "false"
run_and_parse "DART+90%子采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.9 "true" 0.20 "false" "false"
run_and_parse "DART+无子采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

###############################################################################
# 10. DART最优配置探索
###############################################################################
echo "10. DART最优配置探索..."
run_and_parse "保守DART配置" "$DATA_PATH" "squared" 40 0.08 4 2 "mse" "exhaustive" 0.9 "true" 0.15 "false" "false"
run_and_parse "激进DART配置" "$DATA_PATH" "squared" 35 0.15 6 1 "mse" "exhaustive" 0.8 "true" 0.35 "false" "false"
run_and_parse "平衡DART配置" "$DATA_PATH" "squared" 35 0.1 4 1 "mse" "exhaustive" 0.85 "true" 0.25 "false" "false"
run_and_parse "鲁棒DART配置" "$DATA_PATH" "huber" 35 0.1 4 3 "mse" "exhaustive" 0.9 "true" 0.20 "false" "false"

echo ""
echo "=== GBRT DART 单线程测试完成 ==="
echo "详细结果保存在: $OUTFILE"
echo ""
echo "DART测试要点:"
echo "1. DART 0%应该等同于标准GBRT"
echo "2. 适当的丢弃率(10%-30%)通常效果最佳"
echo "3. 权重归一化的影响因数据而异"
echo "4. 预测模式选择影响最终性能"