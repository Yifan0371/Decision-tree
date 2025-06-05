#!/bin/bash
set -euo pipefail

# =============================================================================
# script/mpi_bagging/test_mpi_bagging_strong_scaling.sh
#
# MPI Bagging Strong-scaling test – 固定总工作量，测试不同MPI进程数下的性能
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

# 2) 固定参数（Strong Scaling - 固定总工作量）
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA" ]]; then
  echo "错误：找不到数据文件 $DATA"
  exit 1
fi

# Strong Scaling参数 - 总工作量固定
TOTAL_TREES=80              # 固定总树数
SAMPLE_RATIO=1.0
MAX_DEPTH=12
MIN_LEAF=2
CRITERION="mse"
FINDER="histogram_ew"       # 使用较快的分裂方法
PRUNER="none"
PRUNER_PARAM=0.01
SEED=42

# 3) MPI进程数列表生成
max_procs=$(nproc)          # 获取物理核心数
mpi_procs=(1)               # 从1个进程开始

# 生成2的幂次进程数: 1, 2, 4, 8, ...
current=1
while (( current * 2 <= max_procs )); do
    current=$((current * 2))
    mpi_procs+=($current)
done

# 如果最大核心数不是2的幂，添加最大核心数
if (( mpi_procs[-1] != max_procs )); then
    mpi_procs+=($max_procs)
fi

echo "检测到物理核心数: $max_procs"
echo "将测试MPI进程数: ${mpi_procs[*]}"

# 4) 设置OpenMP线程数（每个MPI进程使用所有可用线程）
# 在Strong Scaling中，通常每个MPI进程使用1个线程，或者根据进程数调整
export OMP_NUM_THREADS=1    # 纯MPI模式
# 或者使用混合模式：export OMP_NUM_THREADS=$((max_procs / 当前MPI进程数))

# 5) 打印测试配置
echo "==============================================="
echo "    MPI Bagging Strong Scaling Test          "
echo "==============================================="
echo "测试类型: Strong Scaling (固定总工作量)"
echo "固定参数:"
echo "  总树数: $TOTAL_TREES (固定)"
echo "  采样率: $SAMPLE_RATIO"
echo "  最大深度: $MAX_DEPTH"
echo "  最小叶子: $MIN_LEAF"
echo "  分裂准则: $CRITERION"
echo "  分裂方法: $FINDER"
echo "  数据文件: $(basename "$DATA")"
echo "  OpenMP线程/进程: $OMP_NUM_THREADS"
echo ""
echo "MPI进程 | 每进程树数 | 总时间(ms) | 测试MSE    | 测试MAE    | 加速比  | 效率(%) | 负载均衡(%)"
echo "--------|-------------|------------|------------|------------|---------|---------|----------"

# 记录基准性能（单进程）
baseline_time=0

for np in "${mpi_procs[@]}"; do
    trees_per_proc=$((TOTAL_TREES / np))
    remainder=$((TOTAL_TREES % np))
    
    # 检查是否所有进程都有工作
    if (( trees_per_proc == 0 )); then
        echo "警告: 进程数($np)大于总树数($TOTAL_TREES)，跳过此配置"
        continue
    fi
    
    # 设置环境变量
    export MPI_BAGGING_VERBOSE=1
    
    # 记录开始时间
    start_ts=$(date +%s%3N)
    
    # 运行MPI Bagging
    output=$(mpirun --oversubscribe -np $np "$EXECUTABLE" \
        "$DATA" $TOTAL_TREES $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED 2>/dev/null)
    
    end_ts=$(date +%s%3N)
    elapsed=$((end_ts - start_ts))
    
    # 解析输出结果
    test_mse=$(echo "$output" | grep -E "测试 MSE:" | sed -n 's/.*测试 MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(echo "$output" | grep -E "测试 MAE:" | sed -n 's/.*测试 MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    load_balance=$(echo "$output" | grep -E "负载平衡效率:" | sed -n 's/.*负载平衡效率: *\([0-9.-]*\)%.*/\1/p' | tail -1)
    
    # 处理解析失败的情况
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"  
    [[ -z "$load_balance" ]] && load_balance="N/A"
    
    # 计算性能指标
    if (( np == 1 )); then
        baseline_time=$elapsed
        speedup="1.00"
        efficiency="100.0"
    else
        if (( baseline_time > 0 && elapsed > 0 )); then
            speedup=$(echo "scale=2; $baseline_time / $elapsed" | bc -l)
            efficiency=$(echo "scale=1; ($baseline_time / $elapsed) * 100 / $np" | bc -l)
        else
            speedup="N/A"
            efficiency="N/A"
        fi
    fi
    
    # 显示每进程树数分配详情
    if (( remainder > 0 )); then
        trees_display="${trees_per_proc}+${remainder}"
    else
        trees_display="$trees_per_proc"
    fi
    
    printf "%7d | %11s | %10d | %-10s | %-10s | %7s | %7s | %s\n" \
           "$np" "$trees_display" "$elapsed" "$test_mse" "$test_mae" "$speedup" "$efficiency" "$load_balance"
    
    # 实时显示进程级统计（如果可用）
    if [[ "$output" == *"进程性能统计"* ]]; then
        echo "    进程详情:"
        echo "$output" | grep -A 10 "进程性能统计" | grep "进程 [0-9]" | head -5 | \
        while read line; do
            echo "      $line"
        done
    fi
done

echo ""
echo "==============================================="
echo "Strong Scaling 分析:"
echo "- 理想情况: 进程数翻倍，运行时间减半"
echo "- 加速比 = 单进程时间 / 当前时间"  
echo "- 效率 = 加速比 / 进程数 × 100%"
echo "- 负载均衡 = 最快进程时间 / 最慢进程时间"
echo ""
echo "性能分析建议:"
if (( baseline_time > 0 )); then
    final_procs=${mpi_procs[-1]}
    if command -v bc &> /dev/null; then
        final_efficiency=$(echo "scale=1; ($baseline_time / $elapsed) * 100 / $final_procs" | bc -l 2>/dev/null || echo "N/A")
        echo "- 最大进程数($final_procs)效率: ${final_efficiency}%"
        
        # 根据效率给出建议
        if [[ "$final_efficiency" != "N/A" ]] && (( $(echo "$final_efficiency >= 70" | bc -l) )); then
            echo "- 扩展性良好，可考虑使用更多进程"
        elif [[ "$final_efficiency" != "N/A" ]] && (( $(echo "$final_efficiency >= 50" | bc -l) )); then
            echo "- 扩展性中等，当前配置较合理"
        else
            echo "- 扩展性较差，建议减少进程数或增加每进程工作量"
        fi
    fi
fi
echo "- 负载均衡 > 90% 为良好，< 80% 需要优化"
echo "==============================================="

exit 0