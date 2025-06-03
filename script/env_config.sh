#!/bin/bash
# =============================================================================
# script/env_config.sh
# 自动：
#   1) 检测物理 CPU 核心数（排除超线程），并设置 OMP_NUM_THREADS
#   2) 如果未构建过项目，则执行 CMake 与 Make
#
# 支持中英文 lscpu 输出。在其他脚本中 source 本文件，即可获得正确的 OMP_NUM_THREADS，
# 并确保 build/DecisionTreeMain 已经存在。
# =============================================================================

# -------- 1. 检测并设置物理核心数（排除超线程） --------
if ! command -v lscpu &> /dev/null; then
    echo "警告：lscpu 不可用，默认 OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
else
    # 尝试英文字段：Socket(s) 和 Core(s) per socket
    SOCKETS=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}')
    CORES_PS=$(lscpu | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2}')
    # 如果英文字段没取到，再尝试中文字段：座： 和 每个座的核数：
    if [[ -z "$SOCKETS" || -z "$CORES_PS" ]]; then
        SOCKETS=$(lscpu | awk -F: '/^ *座：/ {gsub(/ /,"",$2); print $2}')
        CORES_PS=$(lscpu | awk -F: '/^ *每个座的核数：/ {gsub(/ /,"",$2); print $2}')
    fi
    # 如果都能取到，就算物理核数
    if [[ -n "$SOCKETS" && -n "$CORES_PS" ]]; then
        PHYS=$(( SOCKETS * CORES_PS ))
    else
        # 否则退回到逻辑核数/2 作为近似
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
    # 最少 1 核
    if [[ -z "$PHYS" || "$PHYS" -lt 1 ]]; then
        PHYS=1
    fi
    export OMP_NUM_THREADS=$PHYS
    echo "物理核心数 (排除超线程)：$PHYS，已设置 OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi
# -------- 2. 自动构建：CMake && Make --------
# 原来是 “/../..”，但那会把路径算到 Decision-tree 上一级。改成 “/..” 才能正确指向项目根。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
EXECUTABLE="$BUILD_DIR/DecisionTreeMain"

if [[ ! -f "$EXECUTABLE" ]]; then
    echo "未检测到可执行文件 $EXECUTABLE，即将自动执行 CMake && Make"
    mkdir -p "$BUILD_DIR"
    pushd "$BUILD_DIR" > /dev/null

    # 显式告诉 CMake 源码就在 $ROOT_DIR
    cmake -DCMAKE_BUILD_TYPE=Release "$ROOT_DIR"
    if [[ $? -ne 0 ]]; then
        echo "错误：CMake 失败，退出"
        popd > /dev/null
        exit 1
    fi

    make -j"$OMP_NUM_THREADS"
    if [[ $? -ne 0 ]]; then
        echo "错误：Make 失败，退出"
        popd > /dev/null
        exit 1
    fi

    popd > /dev/null
    echo "自动构建完成：$EXECUTABLE 已生成"
else
    echo "检测到可执行文件：$EXECUTABLE"
fi
