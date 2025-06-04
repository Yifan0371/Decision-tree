#!/bin/bash
# =============================================================================
# script/boosting/gbrt/test_gbrt_single.sh
# GBRT (基础不带DART) 单线程全面测试脚本
# 在项目根目录运行: bash script/boosting/gbrt/test_gbrt_single.sh
# =============================================================================

export OMP_NUM_THREADS=1

PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt"
OUTFILE="$RESULTS_DIR/gbrt_single_results.txt"
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

echo "=== GBRT 单线程全面测试 ==="
echo "数据文件: $DATA_PATH"
echo "结果保存: $OUTFILE"
echo ""

###############################################################################
# 1. 损失函数测试
###############################################################################
echo "1. 损失函数测试..."
run_and_parse "Loss=squared,基准配置" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Loss=huber,鲁棒损失" "$DATA_PATH" "huber" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Loss=absolute,L1损失" "$DATA_PATH" "absolute" 30 0.1 4 1 "mae" "exhaustive" 1.0

###############################################################################
# 2. 迭代次数测试
###############################################################################
echo "2. 迭代次数测试..."
run_and_parse "Iters=10,快速训练" "$DATA_PATH" "squared" 10 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=25,标准训练" "$DATA_PATH" "squared" 25 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=50,充分训练" "$DATA_PATH" "squared" 50 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=100,深度训练" "$DATA_PATH" "squared" 100 0.1 4 1 "mse" "exhaustive" 1.0

###############################################################################
# 3. 学习率测试
###############################################################################
echo "3. 学习率测试..."
run_and_parse "LR=0.01,保守学习" "$DATA_PATH" "squared" 30 0.01 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.05,稳健学习" "$DATA_PATH" "squared" 30 0.05 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.1,标准学习" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.2,快速学习" "$DATA_PATH" "squared" 30 0.2 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.5,激进学习" "$DATA_PATH" "squared" 30 0.5 4 1 "mse" "exhaustive" 1.0

###############################################################################
# 4. 树深度测试
###############################################################################
echo "4. 树深度测试..."
run_and_parse "Depth=2,浅树" "$DATA_PATH" "squared" 30 0.1 2 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=3,平衡树" "$DATA_PATH" "squared" 30 0.1 3 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=4,标准树" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=6,深树" "$DATA_PATH" "squared" 30 0.1 6 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=8,极深树" "$DATA_PATH" "squared" 30 0.1 8 1 "mse" "exhaustive" 1.0

###############################################################################
# 5. 最小叶子样本测试
###############################################################################
echo "5. 最小叶子样本测试..."
run_and_parse "MinLeaf=1,精细分割" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=2,标准分割" "$DATA_PATH" "squared" 30 0.1 4 2 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=5,稳健分割" "$DATA_PATH" "squared" 30 0.1 4 5 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=10,保守分割" "$DATA_PATH" "squared" 30 0.1 4 10 "mse" "exhaustive" 1.0

###############################################################################
# 6. 分割方法测试
###############################################################################
echo "6. 分割方法测试..."
run_and_parse "Split=exhaustive,穷举" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Split=random,随机" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "random" 1.0
run_and_parse "Split=histogram_ew,等宽直方图" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_ew" 1.0
run_and_parse "Split=histogram_eq,等频直方图" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_eq" 1.0

###############################################################################
# 7. 子采样测试
###############################################################################
echo "7. 子采样测试..."
run_and_parse "Subsample=0.5,50%采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.5
run_and_parse "Subsample=0.7,70%采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.7
run_and_parse "Subsample=0.8,80%采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.8
run_and_parse "Subsample=1.0,无采样" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0

###############################################################################
# 8. 组合优化测试
###############################################################################
echo "8. 组合优化测试..."
run_and_parse "最优组合1:深度训练" "$DATA_PATH" "squared" 50 0.1 6 1 "mse" "exhaustive" 0.8
run_and_parse "最优组合2:鲁棒训练" "$DATA_PATH" "huber" 50 0.05 4 2 "mse" "exhaustive" 0.9
run_and_parse "快速组合:原型开发" "$DATA_PATH" "squared" 20 0.2 3 1 "mse" "random" 1.0
run_and_parse "平衡组合:生产环境" "$DATA_PATH" "squared" 30 0.15 4 2 "mse" "histogram_ew" 0.85

echo ""
echo "=== GBRT 单线程测试完成 ==="
echo "详细结果保存在: $OUTFILE"