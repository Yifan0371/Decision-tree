#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/weak_scaling_comparison.sh
# 
# 弱扩展性测试：对比MPI+OpenMP混合版本 vs 纯OpenMP版本
# 保持每个处理器的工作量恒定，增加处理器数量的同时按比例增加问题规模
# =============================================================================

# 项目根路径
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

# 确保在build目录中运行（与成功示例一致）
BUILD_DIR="$PROJECT_ROOT/build"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    exit 1
fi

echo "Changing to build directory: $BUILD_DIR"
cd "$BUILD_DIR"

# 可执行文件路径（相对于build目录）
MPI_EXECUTABLE="./MPIBaggingMain"
OPENMP_EXECUTABLE="./DecisionTreeMain"

# 检查可执行文件
if [[ ! -x "$MPI_EXECUTABLE" ]]; then
    echo "ERROR: MPI executable not found: $MPI_EXECUTABLE"
    echo "Please build with: cmake -DENABLE_MPI=ON .. && make"
    exit 1
fi

if [[ ! -x "$OPENMP_EXECUTABLE" ]]; then
    echo "ERROR: OpenMP executable not found: $OPENMP_EXECUTABLE"
    echo "Please build the project first"
    exit 1
fi

# 数据路径 - 使用相对路径（与成功示例一致）
DATA_PATH="../data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# 输出目录（回到项目根目录的script/bagging）
RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

# 结果文件
MPI_RESULTS="$RESULTS_DIR/weak_scaling_mpi_results.csv"
OPENMP_RESULTS="$RESULTS_DIR/weak_scaling_openmp_results.csv"
COMPARISON_RESULTS="$RESULTS_DIR/weak_scaling_comparison.csv"

# 弱扩展性基础参数
BASE_TREES_PER_CORE=25       # 每个核心的基础树数量
FIXED_SAMPLE_RATIO=1.0       # 固定采样率
FIXED_MAX_DEPTH=10           # 固定最大深度
FIXED_MIN_SAMPLES_LEAF=2     # 固定最小叶节点样本数
FIXED_CRITERION="mse"        # 固定准则
FIXED_SPLIT_METHOD="random"  # 固定分裂方法（与成功示例一致）
FIXED_PRUNER_TYPE="none"     # 固定剪枝类型
# 注意：MPI版本只需要8个参数

# CPU架构相关配置
PHYSICAL_CORES=36  # 2 sockets * 18 cores/socket
MAX_CORES=$PHYSICAL_CORES

# 测试配置：核心数序列（弱扩展性测试）
TEST_CONFIGS=(1 2 4 6 9 12 18 24 36)

echo "=========================================="
echo "     Weak Scaling Comparison Test        "
echo "=========================================="
echo "CPU Architecture: Intel Xeon E5-2699 v3"
echo "Physical Cores: $PHYSICAL_CORES"
echo "Test Date: $(date)"
echo ""
echo "Weak Scaling Strategy:"
echo "  Base workload: $BASE_TREES_PER_CORE trees per core"
echo "  Scaling: Trees = Cores × $BASE_TREES_PER_CORE"
echo ""
echo "Fixed Parameters:"
echo "  Sample Ratio: $FIXED_SAMPLE_RATIO"
echo "  Max Depth: $FIXED_MAX_DEPTH"
echo "  Min Samples Leaf: $FIXED_MIN_SAMPLES_LEAF"
echo "  Criterion: $FIXED_CRITERION"
echo "  Split Method: $FIXED_SPLIT_METHOD"
echo "  Data: $(basename "$DATA_PATH")"
echo ""

# 创建结果文件头部
{
    echo "# Weak Scaling Test Results - MPI+OpenMP Version"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,MPI_Processes,OpenMP_Threads,Total_Trees,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Efficiency"
} > "$MPI_RESULTS"

{
    echo "# Weak Scaling Test Results - Pure OpenMP Version"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,OpenMP_Threads,Total_Trees,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Efficiency"
} > "$OPENMP_RESULTS"

# 辅助函数：提取结果（与强扩展性脚本相同）
extract_mpi_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    wall_time=$(grep -E "Total time.*including communication" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Max training time across processes" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

extract_openmp_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    wall_time=$(grep -E "Total Time:" "$log_file" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Train Time:" "$log_file" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

# 基准时间（单核性能）
baseline_mpi_time=""
baseline_openmp_time=""

echo "=========================================="
echo "Starting Weak Scaling Tests..."
echo "=========================================="

# 测试不同配置
for cores in "${TEST_CONFIGS[@]}"; do
    # 计算当前配置的树数量（弱扩展性：工作量按核心数线性增长）
    total_trees=$((cores * BASE_TREES_PER_CORE))
    
    echo ""
    echo "Testing configuration: $cores cores, $total_trees trees"
    echo "----------------------------------------"
    
    # === MPI+OpenMP 混合测试 ===
    echo "  [1/2] Testing MPI+OpenMP version..."
    
    # 计算MPI进程数和每进程OpenMP线程数
    # 简化策略：保持与强扩展性测试一致
    if (( cores <= 4 )); then
        mpi_procs=$cores     # 1-4核心：每核心一个进程
        omp_threads=1
    elif (( cores <= 12 )); then
        mpi_procs=$(( (cores + 1) / 2 ))   # 5-12核心：大约每2核心一个进程
        omp_threads=2
    else
        mpi_procs=$(( (cores + 3) / 4 ))   # >12核心：大约每4核心一个进程
        omp_threads=4
    fi
    
    # 确保至少有1个进程，最多不超过cores个进程
    mpi_procs=$(( mpi_procs > 0 ? mpi_procs : 1 ))
    mpi_procs=$(( mpi_procs > cores ? cores : mpi_procs ))
    
    # 设置OpenMP环境变量
    export OMP_NUM_THREADS=$omp_threads
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    
    mpi_log_file="temp_mpi_weak_${cores}_cores.log"
    
    # 运行MPI版本（简化命令，与成功示例一致）
    timeout 900 mpirun -np $mpi_procs \
        "$MPI_EXECUTABLE" \
        "$DATA_PATH" \
        $total_trees \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        > "$mpi_log_file" 2>&1
    
    if [[ $? -eq 0 ]]; then
        mpi_results=$(extract_mpi_results "$mpi_log_file")
        IFS=',' read -r mpi_wall_time mpi_train_time mpi_test_mse mpi_test_mae <<< "$mpi_results"
        
        # 计算弱扩展性效率（理想情况下时间应该保持恒定）
        if [[ -z "$baseline_mpi_time" && "$mpi_wall_time" != "ERROR" ]]; then
            baseline_mpi_time=$mpi_wall_time
            mpi_efficiency="1.00"
        elif [[ "$mpi_wall_time" != "ERROR" && -n "$baseline_mpi_time" ]]; then
            # 弱扩展性效率 = 基准时间 / 当前时间
            mpi_efficiency=$(echo "scale=3; $baseline_mpi_time / $mpi_wall_time" | bc -l)
        else
            mpi_efficiency="ERROR"
        fi
        
        echo "$cores,$mpi_procs,$omp_threads,$total_trees,$mpi_wall_time,$mpi_train_time,$mpi_test_mse,$mpi_test_mae,$mpi_efficiency" >> "$MPI_RESULTS"
        echo "    MPI+OpenMP: ${mpi_procs}P×${omp_threads}T, Trees: $total_trees, Time: ${mpi_wall_time}ms, Efficiency: ${mpi_efficiency}"
    else
        echo "    MPI+OpenMP: FAILED or TIMEOUT"
        echo "$cores,$mpi_procs,$omp_threads,$total_trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR" >> "$MPI_RESULTS"
    fi
    
    rm -f "$mpi_log_file"
    
    # === 纯OpenMP测试 ===
    echo "  [2/2] Testing Pure OpenMP version..."
    
    # 设置OpenMP环境变量
    export OMP_NUM_THREADS=$cores
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    
    openmp_log_file="temp_openmp_weak_${cores}_cores.log"
    
    # 运行OpenMP版本（包含prunerParam和seed）
    timeout 900 "$OPENMP_EXECUTABLE" bagging \
        "$DATA_PATH" \
        $total_trees \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        0.01 \
        42 \
        > "$openmp_log_file" 2>&1
    
    if [[ $? -eq 0 ]]; then
        openmp_results=$(extract_openmp_results "$openmp_log_file")
        IFS=',' read -r openmp_wall_time openmp_train_time openmp_test_mse openmp_test_mae <<< "$openmp_results"
        
        # 计算弱扩展性效率
        if [[ -z "$baseline_openmp_time" && "$openmp_wall_time" != "ERROR" ]]; then
            baseline_openmp_time=$openmp_wall_time
            openmp_efficiency="1.00"
        elif [[ "$openmp_wall_time" != "ERROR" && -n "$baseline_openmp_time" ]]; then
            openmp_efficiency=$(echo "scale=3; $baseline_openmp_time / $openmp_wall_time" | bc -l)
        else
            openmp_efficiency="ERROR"
        fi
        
        echo "$cores,$cores,$total_trees,$openmp_wall_time,$openmp_train_time,$openmp_test_mse,$openmp_test_mae,$openmp_efficiency" >> "$OPENMP_RESULTS"
        echo "    Pure OpenMP: ${cores}T, Trees: $total_trees, Time: ${openmp_wall_time}ms, Efficiency: ${openmp_efficiency}"
    else
        echo "    Pure OpenMP: FAILED or TIMEOUT"
        echo "$cores,$cores,$total_trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR" >> "$OPENMP_RESULTS"
    fi
    
    rm -f "$openmp_log_file"
done

# 生成对比报告
{
    echo "# Weak Scaling Comparison Report"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,Total_Trees,MPI_Time_ms,OpenMP_Time_ms,MPI_Efficiency,OpenMP_Efficiency,Time_Ratio"
} > "$COMPARISON_RESULTS"

echo ""
echo "=========================================="
echo "Weak Scaling Test Results Summary"
echo "=========================================="
echo ""
echo "Cores | Trees | MPI Time (ms) | OpenMP Time (ms) | MPI Efficiency | OpenMP Efficiency | Time Ratio"
echo "------|-------|---------------|------------------|----------------|-------------------|------------"

# 读取结果并生成对比
for cores in "${TEST_CONFIGS[@]}"; do
    total_trees=$((cores * BASE_TREES_PER_CORE))
    
    # 读取MPI结果
    mpi_line=$(grep "^$cores," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r cores_col mpi_procs omp_threads trees_col mpi_time mpi_train mpi_mse mpi_mae mpi_efficiency <<< "$mpi_line"
    else
        mpi_time="N/A"
        mpi_efficiency="N/A"
    fi
    
    # 读取OpenMP结果
    openmp_line=$(grep "^$cores," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r cores_col threads_col trees_col openmp_time openmp_train openmp_mse openmp_mae openmp_efficiency <<< "$openmp_line"
    else
        openmp_time="N/A"
        openmp_efficiency="N/A"
    fi
    
    # 计算时间比值
    if [[ "$mpi_time" != "N/A" && "$mpi_time" != "ERROR" && "$mpi_time" != "TIMEOUT" && \
          "$openmp_time" != "N/A" && "$openmp_time" != "ERROR" && "$openmp_time" != "TIMEOUT" ]]; then
        time_ratio=$(echo "scale=2; $openmp_time / $mpi_time" | bc -l)
    else
        time_ratio="N/A"
    fi
    
    # 输出到对比文件
    echo "$cores,$total_trees,$mpi_time,$openmp_time,$mpi_efficiency,$openmp_efficiency,$time_ratio" >> "$COMPARISON_RESULTS"
    
    # 格式化输出
    printf "%5s | %5s | %13s | %16s | %14s | %17s | %10s\n" \
           "$cores" "$total_trees" "$mpi_time" "$openmp_time" "$mpi_efficiency" "$openmp_efficiency" "$time_ratio"
done

echo ""
echo "=========================================="
echo "Weak Scaling Performance Analysis"
echo "=========================================="

# 分析弱扩展性趋势
echo ""
echo "Efficiency Analysis:"
echo "-------------------"

# 计算平均效率
mpi_efficiencies=()
openmp_efficiencies=()

for cores in "${TEST_CONFIGS[@]}"; do
    if (( cores == 1 )); then continue; fi  # 跳过基准测试
    
    mpi_line=$(grep "^$cores," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r _ _ _ _ _ _ _ _ mpi_eff <<< "$mpi_line"
        if [[ "$mpi_eff" != "ERROR" && "$mpi_eff" != "N/A" ]]; then
            mpi_efficiencies+=("$mpi_eff")
        fi
    fi
    
    openmp_line=$(grep "^$cores," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r _ _ _ _ _ _ _ openmp_eff <<< "$openmp_line"
        if [[ "$openmp_eff" != "ERROR" && "$openmp_eff" != "N/A" ]]; then
            openmp_efficiencies+=("$openmp_eff")
        fi
    fi
done

if [[ ${#mpi_efficiencies[@]} -gt 0 ]]; then
    mpi_avg_eff=$(echo "${mpi_efficiencies[@]}" | tr ' ' '+' | bc -l)
    mpi_avg_eff=$(echo "scale=3; $mpi_avg_eff / ${#mpi_efficiencies[@]}" | bc -l)
    echo "MPI+OpenMP Average Efficiency (cores > 1): $mpi_avg_eff"
fi

if [[ ${#openmp_efficiencies[@]} -gt 0 ]]; then
    openmp_avg_eff=$(echo "${openmp_efficiencies[@]}" | tr ' ' '+' | bc -l)
    openmp_avg_eff=$(echo "scale=3; $openmp_avg_eff / ${#openmp_efficiencies[@]}" | bc -l)
    echo "Pure OpenMP Average Efficiency (cores > 1): $openmp_avg_eff"
fi

echo ""
echo "Results saved to:"
echo "  MPI+OpenMP results: $MPI_RESULTS"
echo "  Pure OpenMP results: $OPENMP_RESULTS"
echo "  Comparison report: $COMPARISON_RESULTS"
echo ""
echo "Weak Scaling Analysis Notes:"
echo "- Efficiency = BaselineTime / CurrentTime"
echo "- Ideal weak scaling: Efficiency close to 1.0 (constant time)"
echo "- Time Ratio = OpenMP_Time / MPI_Time (>1 means MPI is faster)"
echo "- Good weak scaling: Efficiency > 0.8"
echo "- Problem size scales linearly with cores: Trees = Cores × $BASE_TREES_PER_CORE"
echo "=========================================="