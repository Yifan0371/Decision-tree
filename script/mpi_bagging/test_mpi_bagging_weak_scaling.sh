#!/bin/bash
set -euo pipefail

# =============================================================================
# script/mpi_bagging/test_mpi_bagging_weak_scaling.sh
#
# MPI Bagging Weak-scaling test – 固定每进程工作量，测试随进程数增加的扩展性
# =============================================================================

# 1) 项目根目录 & 可执行文件
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

EXECUTABLE="$PROJECT_ROOT/build/MPIBaggingMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "错误：找不到MPI可执行文件 $EXECUTABLE"
  echo "请先构建项目: cd build && make MPIBaggingMain"
  exit 1
fi

# 检查MPI环境
if ! command -v mpirun &> /dev/null; then
    echo "错误：未找到mpirun命令，请安装MPI"
    exit 1
fi

# 2) Weak Scaling参数 - 每进程工作量固定
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA" ]]; then
  echo "错误：找不到数据文件 $DATA"
  exit 1
fi

# 每进程固定参数
TREES_PER_PROCESS=20        # 每个进程固定训练20棵树
SAMPLE_RATIO=1.0
MAX_DEPTH=12
MIN_LEAF=2
CRITERION="mse"
FINDER="histogram_ew"       # 使用较快的分裂方法
PRUNER="none"
PRUNER_PARAM=0.01
SEED=42

# 3) 生成进程数序列和对应的数据集大小
max_procs=$(nproc)
mpi_procs=(1)

# 生成进程数序列: 1, 2, 4, 8, ...
current=1
while (( current * 2 <= max_procs )); do
    current=$((current * 2))
    mpi_procs+=($current)
done

# 添加最大核心数（如果不是2的幂）
if (( mpi_procs[-1] != max_procs )); then
    mpi_procs+=($max_procs)
fi

echo "检测到物理核心数: $max_procs"
echo "将测试MPI进程数: ${mpi_procs[*]}"

# 4) 动态数据集大小生成
# 为Weak Scaling，我们需要根据进程数调整数据集大小
# 获取原始数据集行数
total_rows=$(( $(wc -l < "$DATA") - 1 ))  # 减去头部行
echo "原始数据集行数: $total_rows"

# 计算基准数据大小（单进程时使用的数据行数）
BASE_ROWS_PER_PROCESS=$((total_rows / max_procs))
if (( BASE_ROWS_PER_PROCESS < 1000 )); then
    BASE_ROWS_PER_PROCESS=1000  # 最小数据集大小
fi

echo "每进程基准数据行数: $BASE_ROWS_PER_PROCESS"

# 5) 设置OpenMP配置
# 在Weak Scaling中，通常固定每进程的线程数
export OMP_NUM_THREADS=1    # 纯MPI模式
# 或者: export OMP_NUM_THREADS=2  # 混合模式

# 6) 打印测试配置
echo "==============================================="
echo "    MPI Bagging Weak Scaling Test            "
echo "==============================================="
echo "测试类型: Weak Scaling (固定每进程工作量)"
echo "固定参数:"
echo "  每进程树数: $TREES_PER_PROCESS (固定)"
echo "  每进程数据行: $BASE_ROWS_PER_PROCESS (基准)"
echo "  采样率: $SAMPLE_RATIO"
echo "  最大深度: $MAX_DEPTH"
echo "  最小叶子: $MIN_LEAF"
echo "  分裂准则: $CRITERION"
echo "  分裂方法: $FINDER"
echo "  OpenMP线程/进程: $OMP_NUM_THREADS"
echo ""
echo "MPI进程 | 总树数 | 数据行数 | 训练时间(ms) | 测试MSE    | 测试MAE    | 效率(%) | 内存效率"
echo "--------|--------|----------|--------------|------------|------------|---------|--------"

# 记录基准性能（单进程）
baseline_time=0
baseline_trees=$TREES_PER_PROCESS

for np in "${mpi_procs[@]}"; do
    # 计算当前配置
    total_trees=$((np * TREES_PER_PROCESS))
    data_rows=$((np * BASE_ROWS_PER_PROCESS))
    
    # 确保数据行数不超过原始数据集
    if (( data_rows > total_rows )); then
        data_rows=$total_rows
        echo "警告: 请求的数据行数($data_rows)超过原始数据集，使用全部数据($total_rows)"
    fi
    
    # 创建对应大小的临时数据集
    lines_to_take=$((data_rows + 1))  # +1 for header
    tmpfile="$PROJECT_ROOT/data/data_clean/tmp_mpi_weak_${np}procs_$$.csv"
    head -n "$lines_to_take" "$DATA" > "$tmpfile"
    
    # 设置环境变量
    export MPI_BAGGING_VERBOSE=1
    
    # 记录开始时间
    start_ts=$(date +%s%3N)
    
    # 运行MPI Bagging
    output=$(mpirun --oversubscribe -np $np "$EXECUTABLE" \
        "$tmpfile" $total_trees $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED 2>/dev/null)
    
    end_ts=$(date +%s%3N)
    elapsed=$((end_ts - start_ts))
    
    # 清理临时文件
    rm -f "$tmpfile"
    
    # 解析输出结果
    test_mse=$(echo "$output" | grep -E "测试 MSE:" | sed -n 's/.*测试 MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(echo "$output" | grep -E "测试 MAE:" | sed -n 's/.*测试 MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    load_balance=$(echo "$output" | grep -E "负载平衡效率:" | sed -n 's/.*负载平衡效率: *\([0-9.-]*\)%.*/\1/p' | tail -1)
    
    # 处理解析失败的情况
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    [[ -z "$load_balance" ]] && load_balance="N/A"
    
    # 计算Weak Scaling效率
    if (( np == 1 )); then
        baseline_time=$elapsed
        efficiency="100.0"
        memory_eff="1.0x"
    else
        if (( baseline_time > 0 && elapsed > 0 )); then
            # Weak Scaling效率 = 基准时间 / 当前时间 (理想情况应该接近100%)
            efficiency=$(echo "scale=1; $baseline_time * 100 / $elapsed" | bc -l)
            # 内存效率 = 实际使用数据 / 理论需要数据
            memory_eff=$(echo "scale=1; $data_rows / ($np * $BASE_ROWS_PER_PROCESS)" | bc -l)
        else
            efficiency="N/A"
            memory_eff="N/A"
        fi
    fi
    
    printf "%7d | %6d | %8d | %12d | %-10s | %-10s | %7s | %s\n" \
           "$np" "$total_trees" "$data_rows" "$elapsed" "$test_mse" "$test_mae" "$efficiency" "$memory_eff"
    
    # 显示详细的进程统计（可选）
    if [[ "$load_balance" != "N/A" ]]; then
        printf "        负载均衡: %s%% | " "$load_balance"
        
        # 提取进程时间分布信息
        min_time=$(echo "$output" | grep -E "最小进程时间:" | sed -n 's/.*最小进程时间: *\([0-9.-]*\)ms.*/\1/p')
        max_time=$(echo "$output" | grep -E "最大进程时间:" | sed -n 's/.*最大进程时间: *\([0-9.-]*\)ms.*/\1/p')
        
        if [[ -n "$min_time" && -n "$max_time" ]]; then
            printf "时间范围: %sms - %sms\n" "$min_time" "$max_time"
        else
            printf "\n"
        fi
    fi
done

echo ""
echo "==============================================="
echo "Weak Scaling 分析:"
echo "- 理想情况: 随着进程数增加，单进程处理时间保持恒定"
echo "- 效率 = 单进程基准时间 / 当前时间 × 100%"
echo "- 内存效率 = 实际数据使用 / 理论数据需求"
echo ""
echo "性能指标解读:"
echo "- 效率 > 90%: 优秀的弱扩展性"
echo "- 效率 70-90%: 良好的弱扩展性"  
echo "- 效率 < 70%: 扩展性有限，可能存在瓶颈"
echo ""

# 分析结果并给出建议
if command -v bc &> /dev/null && (( baseline_time > 0 )); then
    final_procs=${mpi_procs[-1]}
    final_efficiency_index=$((${#mpi_procs[@]} - 1))
    
    echo "性能趋势分析:"
    
    # 检查效率下降趋势
    if (( ${#mpi_procs[@]} >= 3 )); then
        mid_procs=${mpi_procs[1]}
        echo "- 进程数从 1 → $mid_procs → $final_procs 的效率变化趋势"
        
        # 检查是否有明显的效率下降
        echo "- 建议的最佳进程数配置: 2-$((final_procs / 2)) (根据效率权衡)"
    fi
    
    echo ""
    echo "资源使用建议:"
    echo "- 当前配置每进程使用 $BASE_ROWS_PER_PROCESS 行数据"
    echo "- 每进程训练 $TREES_PER_PROCESS 棵树"
    echo "- 总计算量随进程数线性增长"
    
    if (( final_procs >= 8 )); then
        echo "- 大规模并行 (≥8进程) 时注意网络通信开销"
    fi
fi

echo "==============================================="

exit 0