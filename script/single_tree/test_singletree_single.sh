#!/bin/bash
# =============================================================================
# script/single_tree/test_singletree.sh
#
# 单线程模式下，对 SingleTree 模块做全量测试（Criterion/Finder/Pruner + 树深度/叶子大小 网格）
# 只输出每次运行的“配置 | MSE | MAE | Train(ms) | Total(ms)”四项，并汇总到：
#   script/single_tree/results_summary.txt
#
# 使用：在项目根目录运行
#   bash script/single_tree/test_singletree.sh
# =============================================================================

export OMP_NUM_THREADS=1

DATA_PATH="data/data_clean/cleaned_data.csv"
EXECUTABLE="build/DecisionTreeMain"
VAL_SPLIT=0.2

OUTFILE="script/single_tree/results_summary.txt"
mkdir -p "$(dirname "$OUTFILE")"
echo "配置 | MSE | MAE | Train(ms) | Total(ms)" > "$OUTFILE"

if [ ! -f "$EXECUTABLE" ]; then
  echo "错误：找不到可执行文件 $EXECUTABLE"
  exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：找不到数据文件 $DATA_PATH"
  exit 1
fi

# --- 帮助函数：运行并解析一行 Summary ---
run_and_parse() {
  local config_desc="$1"       # 比如 "Criterion=mse,Finder=exhaustive,Pruner=none"
  shift
  local args=( "$@" )          # 传给 DecisionTreeMain single 的其余参数

  # 执行程序并捕获输出
  local raw
  raw=$("$EXECUTABLE" single "${args[@]}" 2>&1)

  # 从 raw 中提取关键字段
  # 假设 raw 包含一行 “MSE: <num> | MAE: <num> | Train: <num>ms | Total: <num>ms”
  # grep 该行并提取数字
  local line
  line=$(echo "$raw" | grep -E "MSE:\s*[0-9.+-eE]+\s*\|\s*MAE:\s*[0-9.+-eE]+\s*\|\s*Train:\s*[0-9]+ms\s*\|\s*Total:\s*[0-9]+ms" | tail -1)

  if [[ -z "$line" ]]; then
    # 如果没有找到，写一个 N/A 占位
    echo "${config_desc} | MSE=N/A | MAE=N/A | Train(N/A) | Total(N/A)" >> "$OUTFILE"
  else
    # 提取 MSE、MAE、Train、Total 四个数字部分
    local mse
    local mae
    local train_ms
    local total_ms
    mse=$(echo "$line" | grep -oP "MSE:\s*\K[0-9.+-eE]+")
    mae=$(echo "$line" | grep -oP "MAE:\s*\K[0-9.+-eE]+")
    train_ms=$(echo "$line" | grep -oP "Train:\s*\K[0-9]+")
    total_ms=$(echo "$line" | grep -oP "Total:\s*\K[0-9]+")
    echo "${config_desc} | ${mse} | ${mae} | ${train_ms} | ${total_ms}" >> "$OUTFILE"
  fi
}

###############################################################################
# 1. Criterion + Finder + Pruner 全量
###############################################################################
# 1.1 Criterion 测试（固定 Finder=exhaustive, Pruner=none）
FINDER="exhaustive"; PRUNER="none"; PRUNER_PARAM=0.0
criterions=("mse" "mae" "huber" "quantile:0.5" "quantile:0.25" "quantile:0.75" "logcosh" "poisson")

for crit in "${criterions[@]}"; do
  config="Criterion=${crit},Finder=${FINDER},Pruner=${PRUNER}"
  run_and_parse "$config" \
    "$DATA_PATH" \
    20 5 "$crit" "$FINDER" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT
done

# 1.2 Finder 测试（固定 Criterion=mse, Pruner=none）
CRITERION="mse"; PRUNER="none"; PRUNER_PARAM=0.0
finders=("exhaustive" "random:10" "random:20" "quartile" "histogram_ew:32" "histogram_ew:64" "histogram_eq:32" "histogram_eq:64" "adaptive_ew:sturges" "adaptive_ew:rice" "adaptive_eq")

for finder in "${finders[@]}"; do
  config="Criterion=${CRITERION},Finder=${finder},Pruner=${PRUNER}"
  run_and_parse "$config" \
    "$DATA_PATH" \
    20 5 "$CRITERION" "$finder" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT
done

# 1.3 Pruner 测试（固定 Criterion=mse, Finder=exhaustive）
CRITERION="mse"; FINDER="exhaustive"
# Pruner none
run_and_parse "Criterion=${CRITERION},Finder=${FINDER},Pruner=none" \
  "$DATA_PATH" \
  20 5 "$CRITERION" "$FINDER" "none" 0.0 $VAL_SPLIT

# Pruner mingain
for gain in 0.001 0.005 0.01 0.02 0.05; do
  config="Criterion=${CRITERION},Finder=${FINDER},Pruner=mingain,Param=${gain}"
  run_and_parse "$config" \
    "$DATA_PATH" \
    20 5 "$CRITERION" "$FINDER" "mingain" $gain $VAL_SPLIT
done

# Pruner cost_complexity
for alpha in 0.001 0.005 0.01 0.02 0.05 0.1; do
  config="Criterion=${CRITERION},Finder=${FINDER},Pruner=cost_complexity,Param=${alpha}"
  run_and_parse "$config" \
    "$DATA_PATH" \
    20 5 "$CRITERION" "$FINDER" "cost_complexity" $alpha $VAL_SPLIT
done

# Pruner reduced_error
run_and_parse "Criterion=${CRITERION},Finder=${FINDER},Pruner=reduced_error" \
  "$DATA_PATH" \
  20 5 "$CRITERION" "$FINDER" "reduced_error" 0.0 $VAL_SPLIT

###############################################################################
# 2. 树深度 (maxDepth) 网格（固定 Criterion=mse, Finder=exhaustive, Pruner=none）
###############################################################################
CRITERION="mse"; FINDER="exhaustive"; PRUNER="none"; PRUNER_PARAM=0.0
for d in 2 4 6 8 10 12 15 20; do
  config="Criterion=${CRITERION},Finder=${FINDER},Pruner=${PRUNER},maxDepth=${d},minLeaf=5"
  run_and_parse "$config" \
    "$DATA_PATH" \
    $d 5 "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT
done

###############################################################################
# 3. 叶子最小样本 (minSamplesLeaf) 网格（固定 Criterion=mse, Finder=exhaustive, Pruner=none, maxDepth=20）
###############################################################################
CRITERION="mse"; FINDER="exhaustive"; PRUNER="none"; PRUNER_PARAM=0.0
for leaf in 1 2 5 10 20 50; do
  config="Criterion=${CRITERION},Finder=${FINDER},Pruner=${PRUNER},maxDepth=20,minLeaf=${leaf}"
  run_and_parse "$config" \
    "$DATA_PATH" \
    20 $leaf "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT
done

echo "全部测试结束，结果保存在 $OUTFILE"
