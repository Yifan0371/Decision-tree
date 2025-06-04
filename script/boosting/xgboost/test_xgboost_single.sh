#!/bin/bash
# =============================================================================
# script/boosting/xgboost/test_xgboost_single.sh
# XGBoost 单线程全面测试脚本
# 在项目根目录运行: bash script/boosting/xgboost/test_xgboost_single.sh
# =============================================================================

export OMP_NUM_THREADS=1

PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/XGBoostMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/xgboost"
OUTFILE="$RESULTS_DIR/xgboost_single_results.txt"
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

echo "=== XGBoost 单线程全面测试 ==="
echo "数据文件: $DATA_PATH"
echo "结果保存: $OUTFILE"
echo ""

###############################################################################
# 1. 目标函数测试
###############################################################################
echo "1. 目标函数测试..."
run_and_parse "Objective=reg:squarederror,基准" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Objective=reg:logistic,逻辑回归" --data "$DATA_PATH" --objective "reg:logistic" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 1

###############################################################################
# 2. 迭代次数测试
###############################################################################
echo "2. 迭代次数测试..."
run_and_parse "Rounds=10,快速训练" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 10 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Rounds=25,标准训练" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 25 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Rounds=50,充分训练" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 50 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Rounds=100,深度训练" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 100 --eta 0.3 --max-depth 6 --min-child-weight 1

###############################################################################
# 3. 学习率测试
###############################################################################
echo "3. 学习率测试..."
run_and_parse "Eta=0.01,保守学习" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.01 --max-depth 6 --min-child-weight 1
run_and_parse "Eta=0.05,稳健学习" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.05 --max-depth 6 --min-child-weight 1
run_and_parse "Eta=0.1,中等学习" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.1 --max-depth 6 --min-child-weight 1
run_and_parse "Eta=0.3,标准学习" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Eta=0.5,快速学习" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.5 --max-depth 6 --min-child-weight 1

###############################################################################
# 4. 最大深度测试
###############################################################################
echo "4. 最大深度测试..."
run_and_parse "Depth=2,浅树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 2 --min-child-weight 1
run_and_parse "Depth=3,平衡树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 3 --min-child-weight 1
run_and_parse "Depth=4,中等树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 4 --min-child-weight 1
run_and_parse "Depth=6,标准树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "Depth=8,深树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 8 --min-child-weight 1
run_and_parse "Depth=10,极深树" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 10 --min-child-weight 1

###############################################################################
# 5. 最小子节点权重测试
###############################################################################
echo "5. 最小子节点权重测试..."
run_and_parse "MinChildWeight=1,标准" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 1
run_and_parse "MinChildWeight=2,稳健" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 2
run_and_parse "MinChildWeight=5,保守" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 5
run_and_parse "MinChildWeight=10,极保守" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --min-child-weight 10

###############################################################################
# 6. L2正则化参数测试
###############################################################################
echo "6. L2正则化测试..."
run_and_parse "Lambda=0.1,弱正则" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --lambda 0.1
run_and_parse "Lambda=0.5,轻度正则" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --lambda 0.5
run_and_parse "Lambda=1.0,标准正则" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --lambda 1.0
run_and_parse "Lambda=2.0,中度正则" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --lambda 2.0
run_and_parse "Lambda=5.0,强正则" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --lambda 5.0

###############################################################################
# 7. 最小分裂损失测试
###############################################################################
echo "7. 最小分裂损失测试..."
run_and_parse "Gamma=0.0,无约束" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --gamma 0.0
run_and_parse "Gamma=0.1,轻度约束" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --gamma 0.1
run_and_parse "Gamma=0.5,中度约束" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --gamma 0.5
run_and_parse "Gamma=1.0,强约束" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --gamma 1.0

###############################################################################
# 8. 子采样率测试
###############################################################################
echo "8. 子采样率测试..."
run_and_parse "Subsample=0.5,50%采样" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --subsample 0.5
run_and_parse "Subsample=0.7,70%采样" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --subsample 0.7
run_and_parse "Subsample=0.8,80%采样" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --subsample 0.8
run_and_parse "Subsample=1.0,无采样" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --subsample 1.0

###############################################################################
# 9. 列采样率测试
###############################################################################
echo "9. 列采样率测试..."
run_and_parse "ColSample=0.5,50%特征" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --colsample-bytree 0.5
run_and_parse "ColSample=0.7,70%特征" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --colsample-bytree 0.7
run_and_parse "ColSample=0.8,80%特征" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --colsample-bytree 0.8
run_and_parse "ColSample=1.0,全特征" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.3 --max-depth 6 --colsample-bytree 1.0

###############################################################################
# 10. 组合优化测试
###############################################################################
echo "10. 组合优化测试..."
run_and_parse "保守配置:防过拟合" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 50 --eta 0.1 --max-depth 4 --min-child-weight 5 --lambda 5.0 --gamma 1.0 --subsample 0.8
run_and_parse "平衡配置:标准场景" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 100 --eta 0.3 --max-depth 6 --min-child-weight 1 --lambda 1.0 --gamma 0.0
run_and_parse "激进配置:快速拟合" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 200 --eta 0.5 --max-depth 8 --min-child-weight 1 --lambda 0.1 --gamma 0.0
run_and_parse "高正则化:小数据集" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 30 --eta 0.1 --max-depth 3 --min-child-weight 10 --lambda 10.0 --gamma 2.0
run_and_parse "快速训练:原型开发" --data "$DATA_PATH" --objective "reg:squarederror" --num-rounds 20 --eta 0.5 --max-depth 3 --min-child-weight 1 --lambda 0.5

echo ""
echo "=== XGBoost 单线程测试完成 ==="
echo "详细结果保存在: $OUTFILE"
echo ""
echo "XGBoost测试要点:"
echo "1. eta(学习率): 0.1-0.3 通常效果最佳"
echo "2. max_depth: 6-8 适合大多数场景"
echo "3. lambda: L2正则化防止过拟合"
echo "4. gamma: 控制分裂的保守程度"
echo "5. subsample: 子采样提升泛化能力"