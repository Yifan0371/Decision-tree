#!/bin/bash
# =============================================================================
# script/env_config.sh - 增强版环境配置
# 功能：
#   1) 检测物理 CPU 核心数（排除超线程），并设置 OMP_NUM_THREADS
#   2) 检测 MPI 环境并设置相关配置
#   3) 提供 MPI + OpenMP 混合并行的最佳配置建议
#   4) 检测系统资源限制和性能参数
# =============================================================================

# -------- 1. 检测并设置物理核心数（排除超线程） --------
detect_physical_cores() {
    local PHYS=1
    
    if ! command -v lscpu &> /dev/null; then
        echo "警告：lscpu 不可用，默认物理核心数=1"
        return 1
    fi
    
    # 先尝试英文字段：Socket(s) 和 Core(s) per socket
    local SOCKETS=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}')
    local CORES_PS=$(lscpu | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2}')
    
    # 如果英文字段没有取到，就尝试中文字段：座： 和 每个座的核数：
    if [[ -z "$SOCKETS" || -z "$CORES_PS" ]]; then
        SOCKETS=$(lscpu | awk -F: '/^ *座：/ {gsub(/ /,"",$2); print $2}')
        CORES_PS=$(lscpu | awk -F: '/^ *每个座的核数：/ {gsub(/ /,"",$2); print $2}')
    fi

    # 依据获取结果计算物理核数
    if [[ -n "$SOCKETS" && -n "$CORES_PS" ]]; then
        PHYS=$(( SOCKETS * CORES_PS ))
    else
        # 英文/中文都没拿到，就退回看总逻辑核数再除以 2
        local LOGICAL=$(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/ /,"",$2); print $2}')
        if [[ -z "$LOGICAL" ]]; then
            LOGICAL=$(lscpu | awk -F: '/^CPU：/ {gsub(/ /,"",$2); print $2}')
            if [[ -z "$LOGICAL" ]]; then
                LOGICAL=$(lscpu | awk -F: '/^CPU:/ {gsub(/ /,"",$2); print $2}')
            fi
        fi
        if [[ -n "$LOGICAL" ]]; then
            PHYS=$(( LOGICAL / 2 ))
        fi
    fi

    # 至少保证为 1 核
    if [[ -z "$PHYS" || "$PHYS" -lt 1 ]]; then
        PHYS=1
    fi
    
    echo "$PHYS"
}

# 获取物理核心数
PHYSICAL_CORES=$(detect_physical_cores)
export OMP_NUM_THREADS=$PHYSICAL_CORES

echo "=== 系统资源检测 ==="
echo "物理核心数 (排除超线程): $PHYSICAL_CORES"
echo "已设置 OMP_NUM_THREADS=$OMP_NUM_THREADS"

# -------- 2. MPI 环境检测 --------
detect_mpi_environment() {
    echo ""
    echo "=== MPI 环境检测 ==="
    
    # 检测 MPI 实现
    local MPI_IMPL="未知"
    local MPI_VERSION="未知"
    local MPIRUN_CMD=""
    
    # 检测 mpirun/mpiexec
    if command -v mpirun &> /dev/null; then
        MPIRUN_CMD="mpirun"
    elif command -v mpiexec &> /dev/null; then
        MPIRUN_CMD="mpiexec"
    fi
    
    if [[ -n "$MPIRUN_CMD" ]]; then
        echo "MPI 启动命令: $MPIRUN_CMD"
        
        # 尝试检测 MPI 实现类型
        if command -v ompi_info &> /dev/null; then
            MPI_IMPL="OpenMPI"
            MPI_VERSION=$(ompi_info | grep "Open MPI:" | head -1 | awk '{print $3}' || echo "未知版本")
        elif command -v mpichversion &> /dev/null; then
            MPI_IMPL="MPICH"
            MPI_VERSION=$(mpichversion | head -1 || echo "未知版本")
        elif [[ -n "$I_MPI_ROOT" ]]; then
            MPI_IMPL="Intel MPI"
            MPI_VERSION=$(mpirun -version 2>&1 | head -1 | awk '{print $4}' || echo "未知版本")
        else
            # 尝试通过 mpirun --version 获取信息
            local VERSION_OUTPUT=$($MPIRUN_CMD --version 2>&1 | head -1)
            if [[ "$VERSION_OUTPUT" == *"Open MPI"* ]]; then
                MPI_IMPL="OpenMPI"
                MPI_VERSION=$(echo "$VERSION_OUTPUT" | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" | head -1)
            elif [[ "$VERSION_OUTPUT" == *"MPICH"* ]]; then
                MPI_IMPL="MPICH"
                MPI_VERSION=$(echo "$VERSION_OUTPUT" | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" | head -1)
            fi
        fi
        
        echo "MPI 实现: $MPI_IMPL"
        echo "MPI 版本: $MPI_VERSION"
        
        # 设置 MPI 环境变量
        export MPI_IMPLEMENTATION="$MPI_IMPL"
        export MPI_LAUNCHER="$MPIRUN_CMD"
        export MPI_AVAILABLE=1
        
        # 检测支持的最大进程数
        local MAX_PROCS=$PHYSICAL_CORES
        
        # OpenMPI 特定配置
        if [[ "$MPI_IMPL" == "OpenMPI" ]]; then
            echo "OpenMPI 特定配置:"
            echo "  - 启用过载: --oversubscribe"
            echo "  - 建议最大进程数: $MAX_PROCS (物理核心数)"
            export OMPI_MCA_btl_vader_single_copy_mechanism=none  # 避免某些系统上的问题
            
            # 检查是否需要 --oversubscribe
            export MPI_OVERSUBSCRIBE_FLAG="--oversubscribe"
        elif [[ "$MPI_IMPL" == "MPICH" ]]; then
            echo "MPICH 特定配置:"
            echo "  - 建议最大进程数: $MAX_PROCS"
            export MPI_OVERSUBSCRIBE_FLAG=""
        fi
        
        export MPI_MAX_PROCS=$MAX_PROCS
        
    else
        echo "MPI 未安装或不可用"
        export MPI_AVAILABLE=0
        echo "  提示: 安装 MPI 以启用并行功能"
        echo "  Ubuntu/Debian: sudo apt-get install openmpi-bin openmpi-dev"
        echo "  CentOS/RHEL: sudo yum install openmpi openmpi-devel"
        echo "  或使用包管理器安装 mpich"
    fi
}

detect_mpi_environment

# -------- 3. 混合并行配置建议 --------
suggest_hybrid_config() {
    if [[ "${MPI_AVAILABLE:-0}" == "1" ]]; then
        echo ""
        echo "=== MPI + OpenMP 混合并行建议 ==="
        
        local TOTAL_CORES=$PHYSICAL_CORES
        
        echo "可用配置选项:"
        echo ""
        
        # 纯 MPI 配置
        echo "1) 纯 MPI 模式:"
        echo "   MPI 进程数: $TOTAL_CORES, OpenMP 线程数: 1"
        echo "   命令: export OMP_NUM_THREADS=1; $MPI_LAUNCHER -np $TOTAL_CORES your_program"
        echo "   适用: 内存充足，进程间通信开销小的情况"
        echo ""
        
        # 混合模式配置
        if (( TOTAL_CORES >= 4 )); then
            local MPI_PROCS=$((TOTAL_CORES / 2))
            local OMP_THREADS=2
            echo "2) 平衡混合模式:"
            echo "   MPI 进程数: $MPI_PROCS, OpenMP 线程数: $OMP_THREADS"
            echo "   命令: export OMP_NUM_THREADS=$OMP_THREADS; $MPI_LAUNCHER -np $MPI_PROCS your_program"
            echo "   适用: 平衡内存使用和通信开销的通用配置"
            echo ""
        fi
        
        if (( TOTAL_CORES >= 8 )); then
            local MPI_PROCS=$((TOTAL_CORES / 4))
            local OMP_THREADS=4
            echo "3) OpenMP 偏重模式:"
            echo "   MPI 进程数: $MPI_PROCS, OpenMP 线程数: $OMP_THREADS"  
            echo "   命令: export OMP_NUM_THREADS=$OMP_THREADS; $MPI_LAUNCHER -np $MPI_PROCS your_program"
            echo "   适用: 内存密集型应用，减少进程间通信"
            echo ""
        fi
        
        # 测试建议
        echo "测试建议:"
        echo "- Strong Scaling: 固定总工作量，测试 1, 2, 4, ... $TOTAL_CORES 进程"
        echo "- Weak Scaling: 固定每进程工作量，数据量随进程数线性增长"
        echo "- 监控指标: 运行时间、内存使用、负载均衡、通信开销"
        
        # 导出配置供脚本使用
        export HYBRID_PURE_MPI_PROCS=$TOTAL_CORES
        export HYBRID_PURE_MPI_THREADS=1
        export HYBRID_BALANCED_MPI_PROCS=$((TOTAL_CORES >= 4 ? TOTAL_CORES/2 : 1))
        export HYBRID_BALANCED_OMP_THREADS=$((TOTAL_CORES >= 4 ? 2 : TOTAL_CORES))
        export HYBRID_OMP_HEAVY_MPI_PROCS=$((TOTAL_CORES >= 8 ? TOTAL_CORES/4 : 1))
        export HYBRID_OMP_HEAVY_OMP_THREADS=$((TOTAL_CORES >= 8 ? 4 : TOTAL_CORES))
    fi
}

suggest_hybrid_config

# -------- 4. 系统性能参数检测 --------
detect_system_limits() {
    echo ""
    echo "=== 系统限制检测 ==="
    
    # 内存信息
    if command -v free &> /dev/null; then
        local TOTAL_MEM=$(free -m | awk '/^Mem:/ {print $2}')
        echo "总内存: ${TOTAL_MEM}MB"
        export SYSTEM_TOTAL_MEMORY_MB=$TOTAL_MEM
        
        # 估算每进程可用内存
        if [[ "${MPI_AVAILABLE:-0}" == "1" ]]; then
            local MEM_PER_PROC=$((TOTAL_MEM / PHYSICAL_CORES))
            echo "每进程可用内存 (估算): ${MEM_PER_PROC}MB"
            export SYSTEM_MEMORY_PER_PROCESS_MB=$MEM_PER_PROC
            
            if (( MEM_PER_PROC < 500 )); then
                echo "警告: 每进程可用内存较少，建议减少并行度"
            fi
        fi
    fi
    
    # 检查文件描述符限制
    local ULIMIT_N=$(ulimit -n)
    echo "文件描述符限制: $ULIMIT_N"
    if (( ULIMIT_N < 1024 )); then
        echo "警告: 文件描述符限制较低，可能影响大规模并行"
    fi
    
    # 检查进程数限制
    local ULIMIT_U=$(ulimit -u)
    echo "用户进程数限制: $ULIMIT_U"
    
    # 检查栈大小限制
    local ULIMIT_S=$(ulimit -s)
    echo "栈大小限制: ${ULIMIT_S}KB"
    if [[ "$ULIMIT_S" == "unlimited" ]] || (( ULIMIT_S > 8192 )); then
        echo "栈大小: 充足"
    else
        echo "警告: 栈大小可能不足，深度递归时需注意"
    fi
}

detect_system_limits

# -------- 5. 导出关键环境变量供其他脚本使用 --------
echo ""
echo "=== 导出的环境变量 ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "PHYSICAL_CORES=$PHYSICAL_CORES"
if [[ "${MPI_AVAILABLE:-0}" == "1" ]]; then
    echo "MPI_AVAILABLE=$MPI_AVAILABLE"
    echo "MPI_LAUNCHER=$MPI_LAUNCHER"
    echo "MPI_MAX_PROCS=$MPI_MAX_PROCS"
    echo "MPI_IMPLEMENTATION=$MPI_IMPLEMENTATION"
fi

echo ""
echo "=== 环境配置完成 ==="