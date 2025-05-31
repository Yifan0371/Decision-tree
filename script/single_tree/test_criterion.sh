# =============================================================================
# script/single_tree/test_criterion.sh
# =============================================================================
#!/bin/bash

# 测试不同criterion (不剪枝 + ExhaustiveSplitFinder)

# 路径设置（相对于项目根目录）
DATA_PATH="../../data/data_clean/cleaned_data.csv"
EXECUTABLE="../../DecisionTreeMain"  # 或者 "../../build/DecisionTreeMain"

MAX_DEPTH=20
MIN_SAMPLES=5
FINDER="exhaustive"
PRUNER="none"
PRUNER_PARAM=0.0
VAL_SPLIT=0.2

echo "=== 测试不同Criterion ==="
echo "配置: 不剪枝 + ExhaustiveSplitFinder"
echo ""

# 创建结果目录
mkdir -p results_criterion

# 测试不同criterion
criterions=("mse" "mae" "huber" "quantile:0.5" "quantile:0.25" "quantile:0.75" "logcosh" "poisson")

for criterion in "${criterions[@]}"; do
    echo "测试 criterion: $criterion"
    
    $EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$criterion" "$FINDER" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT \
        > "results_criterion/criterion_${criterion//[:.]/_}.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ $criterion 完成"
    else
        echo "✗ $criterion 失败"
    fi
done

echo ""
echo "结果保存在 results_criterion/ 目录"
echo "完成时间: $(date)"