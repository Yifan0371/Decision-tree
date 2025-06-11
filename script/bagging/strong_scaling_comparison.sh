#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/strong_scaling_comparison.sh
# 
# 强扩展性测试：对比MPI+OpenMP混合版本 vs 纯OpenMP版本
# 固定问题规模，增加处理器数量，测量加速比和效率
# =============================================================================

# 项目根路径
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

# 确保在build目录中运行（与你的成功示例一致）
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

# 详细检查可执行文件和环境
echo "=== Environment Check ==="
echo "Checking MPI environment..."
if ! command -v mpirun &> /dev/null; then
    echo "ERROR: mpirun not found in PATH"
    echo "Please install MPI: sudo yum install openmpi openmpi-devel"
    echo "Or load MPI module: module load mpi/openmpi"
    exit 1
fi
echo "✓ mpirun found: $(which mpirun)"

echo "Checking executables..."
if [[ ! -f "$MPI_EXECUTABLE" ]]; then
    echo "ERROR: MPI executable not found: $MPI_EXECUTABLE"
    echo "Please build with: cmake -DENABLE_MPI=ON .. && make"
    echo "Available files in build/:"
    ls -la "$PROJECT_ROOT/build/" 2>/dev/null || echo "Build directory not found"
    exit 1
elif [[ ! -x "$MPI_EXECUTABLE" ]]; then
    echo "WARNING: MPI executable not executable, fixing permissions..."
    chmod +x "$MPI_EXECUTABLE"
fi
echo "✓ MPI executable: $MPI_EXECUTABLE"

if [[ ! -f "$OPENMP_EXECUTABLE" ]]; then
    echo "ERROR: OpenMP executable not found: $OPENMP_EXECUTABLE"
    echo "Please build the project first"
    exit 1
elif [[ ! -x "$OPENMP_EXECUTABLE" ]]; then
    echo "WARNING: OpenMP executable not executable, fixing permissions..."
    chmod +x "$OPENMP_EXECUTABLE"
fi
echo "✓ OpenMP executable: $OPENMP_EXECUTABLE"

# 数据路径 - 使用相对路径（与你的成功示例一致）
DATA_PATH="../data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# 输出目录（回到项目根目录的script/bagging）
RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

# 结果文件
MPI_RESULTS="$RESULTS_DIR/strong_scaling_mpi_results.csv"
OPENMP_RESULTS="$RESULTS_DIR/strong_scaling_openmp_results.csv"
COMPARISON_RESULTS="$RESULTS_DIR/strong_scaling_comparison.csv"

# 固定测试参数（强扩展性：固定问题规模）
FIXED_NUM_TREES=100          # 固定树数量
FIXED_SAMPLE_RATIO=1.0       # 固定采样率
FIXED_MAX_DEPTH=10           # 固定最大深度
FIXED_MIN_SAMPLES_LEAF=2     # 固定最小叶节点样本数
FIXED_CRITERION="mse"        # 固定准则
FIXED_SPLIT_METHOD="random"  # 固定分裂方法（使用random，与你的成功示例一致）
FIXED_PRUNER_TYPE="none"     # 固定剪枝类型
# 注意：根据你的成功示例，MPI版本只需要8个参数，不需要prunerParam和seed

# CPU架构相关配置
PHYSICAL_CORES=36  # 2 sockets * 18 cores/socket
MAX_CORES=$PHYSICAL_CORES

# 测试配置：进程/线程数序列
# 完整的36核心测试序列
TEST_CONFIGS=(1 2 4 6 9 12 15 18 21 24 27 30 33 36)

echo "=========================================="
echo "    Strong Scaling Comparison Test       "
echo "=========================================="
echo "CPU Architecture: Intel Xeon E5-2699 v3"
echo "Physical Cores: $PHYSICAL_CORES (2 sockets × 18 cores/socket)"
echo "Test Date: $(date)"
echo ""
echo "Test Range: 1 to $PHYSICAL_CORES cores"
echo "Test Points: ${TEST_CONFIGS[*]}"
echo ""
echo "Fixed Parameters (Strong Scaling):"
echo "  Trees: $FIXED_NUM_TREES"
echo "  Sample Ratio: $FIXED_SAMPLE_RATIO"
echo "  Max Depth: $FIXED_MAX_DEPTH"
echo "  Min Samples Leaf: $FIXED_MIN_SAMPLES_LEAF"
echo "  Criterion: $FIXED_CRITERION"
echo "  Split Method: $FIXED_SPLIT_METHOD"
echo "  Pruner: $FIXED_PRUNER_TYPE"
echo "  Data: $(basename "$DATA_PATH")"
echo "  Note: MPI version uses 8 parameters, OpenMP version uses 10 parameters"
echo ""
echo "Expected Runtime: 2-3 hours (14 configurations × 2 versions × ~5min each)"
echo ""

# 创建结果文件头部
{
    echo "# Strong Scaling Test Results - MPI+OpenMP Version"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: MPI_Processes,OpenMP_Threads,Total_Cores,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Speedup,Efficiency"
} > "$MPI_RESULTS"

{
    echo "# Strong Scaling Test Results - Pure OpenMP Version"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: OpenMP_Threads,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Speedup,Efficiency"
} > "$OPENMP_RESULTS"

# 辅助函数：提取结果
extract_mpi_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    # 提取时间信息（从MPI输出）
    wall_time=$(grep -E "Total time.*including communication" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Max training time across processes" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    
    # 提取精度信息
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    # 设置默认值
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

extract_openmp_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    # 提取时间信息（从OpenMP输出）
    wall_time=$(grep -E "Total Time:" "$log_file" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Train Time:" "$log_file" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    
    # 提取精度信息
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    # 设置默认值
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
echo "Starting Strong Scaling Tests..."
echo "=========================================="

# 测试不同配置
for config in "${TEST_CONFIGS[@]}"; do
    echo ""
    echo "Testing configuration: $config cores"
    echo "----------------------------------------"
    
    # === MPI+OpenMP 混合测试 ===
    echo "  [1/2] Testing MPI+OpenMP version..."
    
    # 计算MPI进程数和每进程OpenMP线程数
    # 智能分配策略：平衡MPI进程数和OpenMP线程数
    if (( config == 1 )); then
        mpi_procs=1
        omp_threads=1
    elif (( config <= 6 )); then
        mpi_procs=$config    # 小规模：每核心一个进程
        omp_threads=1
    elif (( config <= 18 )); then
        # 中等规模：每2-3个核心一个进程
        mpi_procs=$(( (config + 1) / 2 ))
        omp_threads=2
    else
        # 大规模：每3-4个核心一个进程，利用NUMA优化
        mpi_procs=$(( (config + 2) / 3 ))
        omp_threads=3
        if (( mpi_procs * omp_threads < config )); then
            omp_threads=$(( (config + mpi_procs - 1) / mpi_procs ))
        fi
    fi
    
    # 确保进程数不超过核心数
    mpi_procs=$(( mpi_procs > config ? config : mpi_procs ))
    mpi_procs=$(( mpi_procs < 1 ? 1 : mpi_procs ))
    
    # 设置OpenMP环境变量
    export OMP_NUM_THREADS=$omp_threads
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    
    mpi_log_file="temp_mpi_${config}_cores.log"
    
    # 构建并显示MPI命令（简化版，与你的成功示例一致）
    mpi_cmd="timeout 600 mpirun -np $mpi_procs $MPI_EXECUTABLE $DATA_PATH $FIXED_NUM_TREES $FIXED_SAMPLE_RATIO $FIXED_MAX_DEPTH $FIXED_MIN_SAMPLES_LEAF $FIXED_CRITERION $FIXED_SPLIT_METHOD $FIXED_PRUNER_TYPE"
    echo "    Command: $mpi_cmd"
    echo "    Environment: OMP_NUM_THREADS=$omp_threads"
    echo "    Running..."
    
    # 运行MPI版本（简化命令，与你的成功示例一致）
    timeout 600 mpirun -np $mpi_procs \
        "$MPI_EXECUTABLE" \
        "$DATA_PATH" \
        $FIXED_NUM_TREES \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        > "$mpi_log_file" 2>&1
    
    mpi_exit_code=$?
    echo "    Exit code: $mpi_exit_code"
    
    if [[ $mpi_exit_code -eq 0 ]]; then
        mpi_results=$(extract_mpi_results "$mpi_log_file")
        IFS=',' read -r mpi_wall_time mpi_train_time mpi_test_mse mpi_test_mae <<< "$mpi_results"
        
        if [[ -z "$baseline_mpi_time" && "$mpi_wall_time" != "ERROR" \
              && "$mpi_procs" -eq 1 && "$omp_threads" -eq 1 ]]; then
            baseline_mpi_time=$mpi_wall_time
            mpi_speedup="1.00"
            mpi_efficiency="1.00"
        elif [[ "$mpi_wall_time" != "ERROR" && -n "$baseline_mpi_time" ]]; then
            # 其余配置的加速比 = 单进程单线程时间 / 当前配置时间
            mpi_speedup=$(echo "scale=2; $baseline_mpi_time / $mpi_wall_time" | bc -l)
            # 效率 = 加速比 / 总核心数
            mpi_efficiency=$(echo "scale=3; $mpi_speedup / $config" | bc -l)
        else
            mpi_speedup="ERROR"
            mpi_efficiency="ERROR"
        fi
        echo "$mpi_procs,$omp_threads,$config,$mpi_wall_time,$mpi_train_time,$mpi_test_mse,$mpi_test_mae,$mpi_speedup,$mpi_efficiency" >> "$MPI_RESULTS"
        echo "    MPI+OpenMP: ${mpi_procs}P×${omp_threads}T, Time: ${mpi_wall_time}ms, Speedup: ${mpi_speedup}"
    elif [[ $mpi_exit_code -eq 124 ]]; then
        echo "    MPI+OpenMP: TIMEOUT (>600s)"
        echo "    Last 10 lines of output:"
        tail -10 "$mpi_log_file" 2>/dev/null || echo "    (no output)"
        echo "$mpi_procs,$omp_threads,$config,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR,ERROR" >> "$MPI_RESULTS"
    else
        echo "    MPI+OpenMP: FAILED (exit code: $mpi_exit_code)"
        echo "    Error output:"
        tail -20 "$mpi_log_file" 2>/dev/null || echo "    (no output)"
        echo "$mpi_procs,$omp_threads,$config,FAILED,FAILED,FAILED,FAILED,ERROR,ERROR" >> "$MPI_RESULTS"
    fi
    
    # 保留失败的日志文件用于调试，删除成功的日志文件
    if [[ $mpi_exit_code -eq 0 ]]; then
        rm -f "$mpi_log_file"
    else
        echo "    Keeping log file for debugging: $mpi_log_file"
    fi
    
    # === 纯OpenMP测试 ===
    echo "  [2/2] Testing Pure OpenMP version..."
    
    # 设置OpenMP环境变量
    export OMP_NUM_THREADS=$config
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    
    openmp_log_file="temp_openmp_${config}_cores.log"
    
    # 构建并显示OpenMP命令（包含所有参数）
    openmp_cmd="timeout 600 $OPENMP_EXECUTABLE bagging $DATA_PATH $FIXED_NUM_TREES $FIXED_SAMPLE_RATIO $FIXED_MAX_DEPTH $FIXED_MIN_SAMPLES_LEAF $FIXED_CRITERION $FIXED_SPLIT_METHOD $FIXED_PRUNER_TYPE 0.01 42"
    echo "    Command: $openmp_cmd"
    echo "    Environment: OMP_NUM_THREADS=$config"
    echo "    Running..."
    
    # 运行OpenMP版本（包含prunerParam和seed）
    timeout 600 "$OPENMP_EXECUTABLE" bagging \
        "$DATA_PATH" \
        $FIXED_NUM_TREES \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        0.01 \
        42 \
        > "$openmp_log_file" 2>&1
    
    openmp_exit_code=$?
    echo "    Exit code: $openmp_exit_code"
    
    if [[ $openmp_exit_code -eq 0 ]]; then
        openmp_results=$(extract_openmp_results "$openmp_log_file")
        IFS=',' read -r openmp_wall_time openmp_train_time openmp_test_mse openmp_test_mae <<< "$openmp_results"
        
        # 计算OpenMP版本的加速比和效率（以OpenMP单线程性能为基准）
        if [[ -z "$baseline_openmp_time" && "$openmp_wall_time" != "ERROR" && "$config" -eq 1 ]]; then
            baseline_openmp_time=$openmp_wall_time
            openmp_speedup="1.00"
            openmp_efficiency="1.00"
        elif [[ "$openmp_wall_time" != "ERROR" && -n "$baseline_openmp_time" ]]; then
            openmp_speedup=$(echo "scale=2; $baseline_openmp_time / $openmp_wall_time" | bc -l)
            openmp_efficiency=$(echo "scale=3; $openmp_speedup / $config" | bc -l)
        else
            openmp_speedup="ERROR"
            openmp_efficiency="ERROR"
        fi
        
        echo "$config,$openmp_wall_time,$openmp_train_time,$openmp_test_mse,$openmp_test_mae,$openmp_speedup,$openmp_efficiency" >> "$OPENMP_RESULTS"
        echo "    Pure OpenMP: ${config}T, Time: ${openmp_wall_time}ms, Speedup: ${openmp_speedup}"
    elif [[ $openmp_exit_code -eq 124 ]]; then
        echo "    Pure OpenMP: TIMEOUT (>600s)"
        echo "    Last 10 lines of output:"
        tail -10 "$openmp_log_file" 2>/dev/null || echo "    (no output)"
        echo "$config,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR,ERROR" >> "$OPENMP_RESULTS"
    else
        echo "    Pure OpenMP: FAILED (exit code: $openmp_exit_code)"
        echo "    Error output:"
        tail -20 "$openmp_log_file" 2>/dev/null || echo "    (no output)"
        echo "$config,FAILED,FAILED,FAILED,FAILED,ERROR,ERROR" >> "$OPENMP_RESULTS"
    fi
    
    # 保留失败的日志文件用于调试，删除成功的日志文件
    if [[ $openmp_exit_code -eq 0 ]]; then
        rm -f "$openmp_log_file"
    else
        echo "    Keeping log file for debugging: $openmp_log_file"
    fi
done

# 生成对比报告
{
    echo "# Strong Scaling Comparison Report"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: Cores,MPI_Config,OpenMP_Config,MPI_Time_ms,OpenMP_Time_ms,MPI_Speedup,OpenMP_Speedup,MPI_Efficiency,OpenMP_Efficiency,Relative_Performance"
} > "$COMPARISON_RESULTS"

echo ""
echo "=========================================="
echo "Strong Scaling Test Results Summary"
echo "=========================================="
echo ""
echo "Cores | MPI Config | OpenMP Config | MPI Time | OpenMP Time | MPI Speedup | OpenMP Speedup | MPI Efficiency | OpenMP Efficiency"
echo "------|------------|---------------|----------|-------------|-------------|----------------|----------------|------------------"

# 读取结果并生成对比
for config in "${TEST_CONFIGS[@]}"; do
    # 读取MPI结果
    mpi_line=$(grep ",$config," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r mpi_procs omp_threads cores mpi_time mpi_train mpi_mse mpi_mae mpi_speedup mpi_eff <<< "$mpi_line"
        mpi_config="${mpi_procs}P×${omp_threads}T"
    else
        mpi_time="N/A"
        mpi_speedup="N/A"
        mpi_eff="N/A"
        mpi_config="N/A"
    fi
    
    # 读取OpenMP结果
    openmp_line=$(grep "^$config," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r threads openmp_time openmp_train openmp_mse openmp_mae openmp_speedup openmp_eff <<< "$openmp_line"
        openmp_config="${threads}T"
    else
        openmp_time="N/A"
        openmp_speedup="N/A"
        openmp_eff="N/A"
        openmp_config="N/A"
    fi
    
    # 计算相对性能
    if [[ "$mpi_time" != "N/A" && "$mpi_time" != "ERROR" && "$mpi_time" != "TIMEOUT" && \
          "$openmp_time" != "N/A" && "$openmp_time" != "ERROR" && "$openmp_time" != "TIMEOUT" ]]; then
        relative_perf=$(echo "scale=2; $openmp_time / $mpi_time" | bc -l)
    else
        relative_perf="N/A"
    fi
    
    # 输出到对比文件
    echo "$config,$mpi_config,$openmp_config,$mpi_time,$openmp_time,$mpi_speedup,$openmp_speedup,$mpi_eff,$openmp_eff,$relative_perf" >> "$COMPARISON_RESULTS"
    
    # 格式化输出
    printf "%5s | %10s | %13s | %8s | %11s | %11s | %14s | %14s | %17s\n" \
           "$config" "$mpi_config" "$openmp_config" "$mpi_time" "$openmp_time" "$mpi_speedup" "$openmp_speedup" "$mpi_eff" "$openmp_eff"
done

echo ""
echo "=========================================="
echo "Results saved to:"
echo "  MPI+OpenMP results: $MPI_RESULTS"
echo "  Pure OpenMP results: $OPENMP_RESULTS"
echo "  Comparison report: $COMPARISON_RESULTS"
echo ""
echo "Analysis Notes:"
echo "- MPI Speedup = MPI_SingleProcess_Time / MPI_Current_Time"
echo "- OpenMP Speedup = OpenMP_SingleThread_Time / OpenMP_Current_Time"
echo "- Efficiency = Speedup / NumberOfCores"
echo "- Relative Performance = OpenMP_Time / MPI_Time (>1 means MPI is faster)"
echo "- Ideal strong scaling: Linear speedup, Efficiency close to 1.0"
echo "- MPI Config: NP×NT means N processes × T threads per process"
echo "=========================================="