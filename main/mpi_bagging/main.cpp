// =============================================================================
// main/mpi_bagging/main.cpp - MPI+OpenMP混合并行Bagging主程序
// =============================================================================
#include "app/MPIBaggingApp.hpp"
#include <mpi.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

void printUsage(const char* programName) {
    std::cout << "用法: " << programName << " <数据文件> [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  <数据文件>            训练数据CSV文件路径" << std::endl;
    std::cout << "  [树数量]              决策树数量 (默认: 20)" << std::endl;
    std::cout << "  [采样率]              Bootstrap采样率 (默认: 1.0)" << std::endl;
    std::cout << "  [最大深度]            树的最大深度 (默认: 10)" << std::endl;
    std::cout << "  [最小叶子样本]        叶子节点最小样本数 (默认: 2)" << std::endl;
    std::cout << "  [分裂准则]            mse|mae|huber|quantile (默认: mse)" << std::endl;
    std::cout << "  [分裂方法]            exhaustive|histogram_ew|random (默认: exhaustive)" << std::endl;
    std::cout << "  [剪枝类型]            none|mingain|cost_complexity (默认: none)" << std::endl;
    std::cout << "  [剪枝参数]            剪枝算法参数 (默认: 0.01)" << std::endl;
    std::cout << "  [随机种子]            随机数种子 (默认: 42)" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  mpirun -np 4 " << programName << " data.csv 100 1.0 15 2 mse exhaustive none 0.01 42" << std::endl;
    std::cout << std::endl;
    std::cout << "环境变量:" << std::endl;
    std::cout << "  OMP_NUM_THREADS       每个MPI进程的OpenMP线程数" << std::endl;
    std::cout << "  MPI_BAGGING_VERBOSE   设置为1启用详细输出" << std::endl;
}

MPIBaggingOptions parseCommandLine(int argc, char** argv) {
    MPIBaggingOptions opts;
    
    // 默认参数
    opts.dataPath = "";
    opts.numTrees = 20;
    opts.sampleRatio = 1.0;
    opts.maxDepth = 10;
    opts.minSamplesLeaf = 2;
    opts.criterion = "mse";
    opts.splitMethod = "exhaustive";
    opts.prunerType = "none";
    opts.prunerParam = 0.01;
    opts.seed = 42;
    opts.verbose = true;
    opts.gatherResults = true;
    
    // 检查环境变量
    const char* verboseEnv = std::getenv("MPI_BAGGING_VERBOSE");
    if (verboseEnv && std::string(verboseEnv) == "0") {
        opts.verbose = false;
    }
    
    if (argc < 2) {
        return opts;  // 返回默认参数，让调用者检查dataPath为空
    }
    
    // 解析命令行参数
    opts.dataPath = argv[1];
    
    if (argc >= 3)  opts.numTrees = std::stoi(argv[2]);
    if (argc >= 4)  opts.sampleRatio = std::stod(argv[3]);
    if (argc >= 5)  opts.maxDepth = std::stoi(argv[4]);
    if (argc >= 6)  opts.minSamplesLeaf = std::stoi(argv[5]);
    if (argc >= 7)  opts.criterion = argv[6];
    if (argc >= 8)  opts.splitMethod = argv[7];
    if (argc >= 9)  opts.prunerType = argv[8];
    if (argc >= 10) opts.prunerParam = std::stod(argv[9]);
    if (argc >= 11) opts.seed = static_cast<uint32_t>(std::stoi(argv[10]));
    
    return opts;
}

void printMPIEnvironmentInfo() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== MPI环境信息 ===" << std::endl;
        std::cout << "MPI进程数: " << size << std::endl;
        
        #ifdef _OPENMP
        std::cout << "OpenMP支持: 启用" << std::endl;
        #pragma omp parallel
        {
            #pragma omp master
            {
                std::cout << "每进程OpenMP线程数: " << omp_get_num_threads() << std::endl;
            }
        }
        std::cout << "总并行度: " << size << " × " << omp_get_max_threads() 
                  << " = " << (size * omp_get_max_threads()) << std::endl;
        #else
        std::cout << "OpenMP支持: 未启用" << std::endl;
        std::cout << "总并行度: " << size << " (仅MPI)" << std::endl;
        #endif
        
        // 显示MPI版本信息
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        std::cout << "MPI版本: " << version << "." << subversion << std::endl;
        std::cout << "===================" << std::endl;
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数（在MPI初始化之前）
    MPIBaggingOptions opts = parseCommandLine(argc, argv);
    
    // 检查必需参数
    if (opts.dataPath.empty()) {
        std::cerr << "错误: 必须指定数据文件路径" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // 检查MPI环境
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "警告: MPI不支持多线程，OpenMP可能无法正常工作" << std::endl;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 显示环境信息和参数
    if (rank == 0) {
        printMPIEnvironmentInfo();
        
        if (opts.verbose) {
            std::cout << "\n=== 训练参数 ===" << std::endl;
            std::cout << "数据文件: " << opts.dataPath << std::endl;
            std::cout << "总树数: " << opts.numTrees << std::endl;
            std::cout << "采样率: " << opts.sampleRatio << std::endl;
            std::cout << "最大深度: " << opts.maxDepth << std::endl;
            std::cout << "最小叶子样本: " << opts.minSamplesLeaf << std::endl;
            std::cout << "分裂准则: " << opts.criterion << std::endl;
            std::cout << "分裂方法: " << opts.splitMethod << std::endl;
            std::cout << "剪枝类型: " << opts.prunerType;
            if (opts.prunerType != "none") {
                std::cout << " (参数: " << opts.prunerParam << ")";
            }
            std::cout << std::endl;
            std::cout << "随机种子: " << opts.seed << std::endl;
            std::cout << "=================" << std::endl;
        }
    }
    
    // 验证参数合理性
    if (opts.numTrees <= 0) {
        if (rank == 0) {
            std::cerr << "错误: 树数量必须大于0" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (opts.numTrees < size) {
        if (rank == 0) {
            std::cerr << "警告: 树数量(" << opts.numTrees 
                      << ")小于进程数(" << size << ")，部分进程将空闲" << std::endl;
        }
    }
    
    try {
        // 运行MPI Bagging训练
        auto result = runMPIBaggingApp(opts);
        
        // 主进程输出最终结果
        if (rank == 0) {
            std::cout << "\n=== 最终结果 ===" << std::endl;
            std::cout << "测试 MSE: " << std::fixed << std::setprecision(6) << result.testMSE << std::endl;
            std::cout << "测试 MAE: " << std::fixed << std::setprecision(6) << result.testMAE << std::endl;
            std::cout << "OOB MSE: " << std::fixed << std::setprecision(6) << result.oobMSE << std::endl;
            std::cout << "总训练时间: " << result.totalTrainTime << " ms" << std::endl;
            std::cout << "平均每树时间: " << std::fixed << std::setprecision(2) 
                      << (result.totalTrainTime / result.totalTrees) << " ms" << std::endl;
            
            // 并行效率分析
            double idealTime = result.totalTrainTime / size;
            double efficiency = (result.minTrainTime / result.maxTrainTime) * 100;
            std::cout << "负载平衡效率: " << std::fixed << std::setprecision(1) 
                      << efficiency << "%" << std::endl;
            
            if (efficiency < 80.0) {
                std::cout << "提示: 负载平衡效率较低，可能需要调整树数量或进程数配比" << std::endl;
            }
            
            std::cout << "=================" << std::endl;
        }
        
        // 所有进程同步等待
        MPI_Barrier(MPI_COMM_WORLD);
        
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "错误: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}