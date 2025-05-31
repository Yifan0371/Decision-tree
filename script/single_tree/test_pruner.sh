
# =============================================================================
# script/single_tree/test_pruner.sh
# =============================================================================
#!/bin/bash

# 测试不同剪枝方法 (ExhaustiveSplitFinder + MSE)

# 路径设置（相对于项目根目录）
DATA_PATH="../../data/data_clean/cleaned_data.csv"
EXECUTABLE="../../DecisionTreeMain"  # 或者 "../../build/DecisionTreeMain"

MAX_DEPTH=20
MIN_SAMPLES=5
CRITERION="mse"
FINDER="exhaustive"
VAL_SPLIT=0.2

echo "=== 测试不同剪枝方法 ==="
echo "配置: ExhaustiveSplitFinder + MSE"
echo ""

# 创建结果目录
mkdir -p results_pruner

# 1. 不剪枝
echo "测试 pruner: none"
$EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$CRITERION" "$FINDER" "none" 0.0 $VAL_SPLIT \
    > "results_pruner/pruner_none.txt" 2>&1
echo "✓ none 完成"

# 2. 预剪枝 - 最小增益
min_gains=(0.001 0.005 0.01 0.02 0.05)
for gain in "${min_gains[@]}"; do
    echo "测试 pruner: mingain $gain"
    $EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$CRITERION" "$FINDER" "mingain" $gain $VAL_SPLIT \
        > "results_pruner/pruner_mingain_${gain}.txt" 2>&1
    echo "✓ mingain $gain 完成"
done

# 3. 后剪枝 - 复杂度剪枝
alphas=(0.001 0.005 0.01 0.02 0.05 0.1)
for alpha in "${alphas[@]}"; do
    echo "测试 pruner: cost_complexity $alpha"
    $EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$CRITERION" "$FINDER" "cost_complexity" $alpha $VAL_SPLIT \
        > "results_pruner/pruner_complexity_${alpha}.txt" 2>&1
    echo "✓ cost_complexity $alpha 完成"
done

# 4. 后剪枝 - 减少错误剪枝
echo "测试 pruner: reduced_error"
$EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$CRITERION" "$FINDER" "reduced_error" 0.0 $VAL_SPLIT \
    > "results_pruner/pruner_reduced_error.txt" 2>&1
echo "✓ reduced_error 完成"

echo ""
echo "结果保存在 results_pruner/ 目录"
echo "完成时间: $(date)"