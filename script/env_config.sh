#!/bin/bash
# =============================================================================
# script/env_config.sh
# 功能：
#   1) 检测物理 CPU 核心数（排除超线程），并设置 OMP_NUM_THREADS
#   2) 不再自动检测或编译可执行文件，其他脚本只需 source 本文件即可获取正确的 OMP_NUM_THREADS
# =============================================================================

# -------- 1. 检测并设置物理核心数（排除超线程） --------
if ! command -v lscpu &> /dev/null; then
    echo "警告：lscpu 不可用，默认 OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
else
    # 先尝试英文字段：Socket(s) 和 Core(s) per socket
    SOCKETS=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}')
    CORES_PS=$(lscpu | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2}')
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
        LOGICAL=$(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/ /,"",$2); print $2}')
        if [[ -z "$LOGICAL" ]]; then
            LOGICAL=$(lscpu | awk -F: '/^CPU：/ {gsub(/ /,"",$2); print $2}')
            if [[ -z "$LOGICAL" ]]; then
                LOGICAL=$(lscpu | awk -F: '/^CPU:/ {gsub(/ /,"",$2); print $2}')
            fi
        fi
        if [[ -n "$LOGICAL" ]]; then
            PHYS=$(( LOGICAL / 2 ))
        else
            PHYS=1
        fi
    fi

    # 至少保证为 1 核
    if [[ -z "$PHYS" || "$PHYS" -lt 1 ]]; then
        PHYS=1
    fi

    export OMP_NUM_THREADS=$PHYS
    echo "物理核心数 (排除超线程)：$PHYS，已设置 OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

# -------- 2. 不做自动构建，交由各模块脚本自行决定可执行的名称与路径 --------
# （如果需要自动构建，可以在各自脚本里明确写：source env_config.sh；然后自行 cmake/make）
