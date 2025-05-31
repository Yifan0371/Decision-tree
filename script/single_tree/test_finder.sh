
# =============================================================================
# script/single_tree/test_finder.sh
# =============================================================================
#!/bin/bash

# 测试不同finder (不剪枝 + MSE)

# 路径设置（相对于项目根目录）
DATA_PATH="../../data/data_clean/cleaned_data.csv"
EXECUTABLE="../../DecisionTreeMain"  # 或者 "../../build/DecisionTreeMain"

MAX_DEPTH=20
MIN_SAMPLES=5
CRITERION="mse"
PRUNER="none"
PRUNER_PARAM=0.0
VAL_SPLIT=0.2

echo "=== 测试不同Finder ==="
echo "配置: 不剪枝 + MSE"
echo ""

# 创建结果目录
mkdir -p results_finder

# 测试不同finder
finders=("exhaustive" "random:10" "random:20" "quartile" "histogram_ew:32" "histogram_ew:64" "histogram_eq:32" "histogram_eq:64" "adaptive_ew:sturges" "adaptive_ew:rice" "adaptive_eq")

for finder in "${finders[@]}"; do
    echo "测试 finder: $finder"
    
    $EXECUTABLE "$DATA_PATH" $MAX_DEPTH $MIN_SAMPLES "$CRITERION" "$finder" "$PRUNER" $PRUNER_PARAM $VAL_SPLIT \
        > "results_finder/finder_${finder//[:.]/_}.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ $finder 完成"
    else
        echo "✗ $finder 失败"
    fi
done

echo ""
echo "结果保存在 results_finder/ 目录"
echo "完成时间: $(date)"