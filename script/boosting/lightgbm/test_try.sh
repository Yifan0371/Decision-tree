#!/bin/bash

# =============================================================================
# script/boosting/lightgbm/test_lightgbm_comprehensive.sh
# LightGBM 全面参数测试脚本
# 在项目根目录运行: bash script/boosting/lightgbm/test_lightgbm_comprehensive.sh
# =============================================================================

# 数据文件相对项目根目录的路径
DATA_PATH="data/data_clean/cleaned_data.csv"
# 可执行文件路径（相对于项目根目录）
EXECUTABLE="build/LightGBMMain"

# 结果文件保存路径
RESULTS_DIR="script/boosting/lightgbm"
RESULTS_FILE="$RESULTS_DIR/lightgbm_comprehensive_results.txt"

# 创建结果目录（如果不存在）
mkdir -p "$RESULTS_DIR"

# 清空之前的结果文件
> "$RESULTS_FILE"

# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 输出函数：同时输出到控制台和文件
output() {
    echo "$1" | tee -a "$RESULTS_FILE"
}

# 仅输出到文件
output_to_file_only() {
    echo "$1" >> "$RESULTS_FILE"
}

# 仅输出到控制台
output_to_console_only() {
    echo "$1"
}

output "=== LightGBM 全面参数测试 ==="
output "开始时间: $START_TIME"
output "数据文件: $DATA_PATH"
output "可执行文件: $EXECUTABLE"
output "结果保存到: $RESULTS_FILE"
output ""

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    output "错误: 找不到可执行文件 $EXECUTABLE"
    output "请先在 build/ 目录下执行 make，确保 LightGBMMain 已编译完成"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    output "错误: 找不到数据文件 $DATA_PATH"
    output "请检查数据文件是否存在，或修改脚本中的 DATA_PATH 指向其他文件"
    exit 1
fi

output "使用数据文件: $DATA_PATH"
output ""

# 从 LightGBMMain 的输出中提取指标（Test MSE、Train MSE、Train Time、树数量）
extract_metrics() {
    local output_text="$1"
    # 提取 Test MSE
    local test_mse=$(echo "$output_text" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    # 提取 Train MSE
    local train_mse=$(echo "$output_text" | grep "Train MSE:" | sed -n 's/.*Train MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    # 提取 Train Time (ms)
    local train_time=$(echo "$output_text" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    # 提取树的数量（Trees: X）
    local trees=$(echo "$output_text" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

    # 处理空值或 NaN
    [ -z "$test_mse" ] || [ "$test_mse" = "-nan" ] && test_mse="ERROR"
    [ -z "$train_mse" ] || [ "$train_mse" = "-nan" ] && train_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"

    echo "$test_mse,$train_mse,$train_time,$trees"
}

# 运行一次测试
# 参数说明：
#   $1：测试描述（字符串）
#   $2：切分方法（如 "histogram_eq:64"、"adaptive_ew:sturges"、"adaptive_eq" 等）
#   $3：迭代轮数（num-iterations）
#   $4：学习率（learning-rate）
#   $5：叶子数（num-leaves）
#   $6：最小数据量（min-data-in-leaf）
run_test() {
    local desc="$1"
    local split_method="$2"
    local num_iter="$3"
    local lr="$4"
    local num_leaves="$5"
    local min_data_leaf="$6"

    # 记录命令开始时间（毫秒）
    local start_time=$(date +%s%3N)

    # 执行 LightGBMMain 命令并捕获输出
    # --quiet：关闭冗余日志，保留关键指标输出
    local output_text=$($EXECUTABLE \
        --data "$DATA_PATH" \
        --split-method "$split_method" \
        --num-iterations "$num_iter" \
        --learning-rate "$lr" \
        --num-leaves "$num_leaves" \
        --min-data-in-leaf "$min_data_leaf" \
        --quiet 2>&1)

    local exit_code=$?

    # 记录结束时间（毫秒）
    local end_time=$(date +%s%3N)
    local wall_time=$((end_time - start_time))

    # 如果程序执行失败，直接输出错误行
    if [ $exit_code -ne 0 ]; then
        # 控制台输出详细信息
        printf "%-25s | %-10s | %-8s | %-8s | %s [ERROR: exit code %d]\n" \
               "$desc" "ERROR" "0ms" "${wall_time}ms" \
               "split=$split_method iters=$num_iter lr=$lr leaves=$num_leaves minleaf=$min_data_leaf" \
               "$exit_code"
        # 文件仅输出简洁信息
        printf "%-25s | %-10s | %-8s | %-8s | ERROR\n" \
               "$desc" "ERROR" "0ms" "${wall_time}ms" >> "$RESULTS_FILE"
        return
    fi

    # 解析输出，得到 test_mse, train_mse, train_time, trees
    local metrics=$(extract_metrics "$output_text")
    local test_mse=$(echo "$metrics" | cut -d',' -f1)
    local train_mse=$(echo "$metrics" | cut -d',' -f2)
    local train_time=$(echo "$metrics" | cut -d',' -f3)
    local trees=$(echo "$metrics" | cut -d',' -f4)

    # 控制台输出详细信息
    printf "%-25s | %-10s | %-8s | %-8s | %-5s | %s\n" \
           "$desc" "$test_mse" "${train_time}ms" "${wall_time}ms" "$trees" \
           "split=$split_method iters=$num_iter lr=$lr leaves=$num_leaves minleaf=$min_data_leaf"

    # 文件输出简洁信息
    printf "%-25s | %-10s | %-8s | %-8s | %-5s\n" \
           "$desc" "$test_mse" "${train_time}ms" "${wall_time}ms" "$trees" >> "$RESULTS_FILE"
}

# 打印表头（控制台 & 文件）
# 控制台表头（含参数配置部分）
printf "%-25s | %-10s | %-8s | %-8s | %-5s | %s\n" \
       "测试描述" "TestMSE" "TrainTime" "WallTime" "Trees" "参数配置"
echo "---------------------------------------------------------------------------------------------"
# 文件表头（简洁）
printf "%-25s | %-10s | %-8s | %-8s | %-5s\n" \
       "测试描述" "TestMSE" "TrainTime" "WallTime" "Trees" >> "$RESULTS_FILE"
echo "----------------------------------------------------------------------------" >> "$RESULTS_FILE"

output ""
output "=== 1. 切分方法测试 ==="
# 自适应等宽（Sturges 规则）
run_test "自适应等宽-Sturges" "adaptive_ew:sturges" 100 0.1 31 20
# 自适应等宽-Rice 规则
run_test "自适应等宽-Rice"    "adaptive_ew:rice"    100 0.1 31 20
# 等频直方图，64 bins
run_test "等频直方图-64"      "histogram_eq:64"     100 0.1 31 20
# 等宽直方图，64 bins
run_test "等宽直方图-64"      "histogram_ew:64"     100 0.1 31 20
# 自适应等频（min-samples-per-bin=10）
run_test "自适应等频-10"      "adaptive_eq"         100 0.1 31 20

output ""
output "=== 2. 迭代次数 & 学习率测试 ==="
# 不同迭代次数与学习率组合
run_test "iters=50, lr=0.05"   "histogram_eq:64"  50  0.05  31 20
run_test "iters=100, lr=0.05"  "histogram_eq:64" 100  0.05  31 20
run_test "iters=100, lr=0.1"   "histogram_eq:64" 100  0.10  31 20
run_test "iters=200, lr=0.1"   "histogram_eq:64" 200  0.10  31 20
run_test "iters=200, lr=0.2"   "histogram_eq:64" 200  0.20  31 20

output ""
output "=== 3. 叶子数 & 最小数据量测试 ==="
# 不同叶子数与最小数据数量组合
run_test "leaves=15, minleaf=10" "histogram_eq:64" 100 0.1  15 10
run_test "leaves=31, minleaf=10" "histogram_eq:64" 100 0.1  31 10
run_test "leaves=63, minleaf=10" "histogram_eq:64" 100 0.1  63 10
run_test "leaves=31, minleaf=5"  "histogram_eq:64" 100 0.1  31  5
run_test "leaves=31, minleaf=20" "histogram_eq:64" 100 0.1  31 20

output ""
output "=== 4. 组合优化测试 ==="
# 将切分方法、迭代次数、学习率、叶子数、最小数据量混合组合
run_test "comb1: adp_ew,100,it=100,lr=0.05,leaves=15,minleaf=10" "adaptive_ew:sturges" 100 0.05 15 10
run_test "comb2: hist_eq,200,it=200,lr=0.1,leaves=31,minleaf=20"   "histogram_eq:64"    200 0.10 31 20
run_test "comb3: adp_eq,150,it=150,lr=0.1,leaves=31,minleaf=15"    "adaptive_eq"        150 0.10 31 15

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")

output ""
output "=== LightGBM 全面测试完成 ==="
output "结束时间: $END_TIME"
output "结果已保存到: $RESULTS_FILE"

# 为了调试，再运行一遍最简单的测试，直接输出到控制台
output_to_console_only ""
output_to_console_only "=== 调试信息 ==="
output_to_console_only "示例：简单跑一次 histogram_eq:64, iters=10, lr=0.1, leaves=31, minleaf=20"
$EXECUTABLE --data "$DATA_PATH" \
            --split-method histogram_eq:64 \
            --num-iterations 10 \
            --learning-rate 0.1 \
            --num-leaves 31 \
            --min-data-in-leaf 20
