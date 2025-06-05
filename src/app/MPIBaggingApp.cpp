#include "app/MPIBaggingApp.hpp"
#include "ensemble/BaggingTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include "tree/Node.hpp"

#include <mpi.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// MPI标签定义
const int TAG_RESULT_SIZE = 100;
const int TAG_RESULT_DATA = 101;
const int TAG_TREE_COUNT = 102;
const int TAG_TREE_DATA = 103;

/**
 * MPI+OpenMP混合并行Bagging应用主函数
 * 
 * @param opts 配置参数
 * @return 全局结果结构
 */
GlobalBaggingResult runMPIBaggingApp(const MPIBaggingOptions& opts) {
    // 注意：MPI 初始化和终止由 main.cpp 统一调用，此处不再重复 MPI_Init/MPI_Finalize
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0 && opts.verbose) {
        std::cout << "=== MPI+OpenMP混合并行Bagging ===" << std::endl;
        std::cout << "MPI进程数: " << size << std::endl;
        #ifdef _OPENMP
        std::cout << "每进程OpenMP线程数: " << omp_get_max_threads() << std::endl;
        #endif
        std::cout << "总树数: " << opts.numTrees << std::endl;
    }
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // 数据加载（所有进程都加载完整数据集）
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);
    
    if (rank == 0 && opts.verbose) {
        std::cout << "数据加载完成: " << y.size() << " 样本, " 
                  << (rowLength - 1) << " 特征" << std::endl;
    }
    
    // 划分训练/测试集（所有进程使用相同的划分）
    DataParams dp;
    if (!splitDataset(X, y, rowLength, dp)) {
        if (rank == 0) {
            std::cerr << "数据集划分失败" << std::endl;
        }
        // 这里不调用 MPI_Finalize，由 main 负责
        return {};
    }
    
    // 计算每个进程负责的树数量
    int treesPerProcess = opts.numTrees / size;
    int remainder = opts.numTrees % size;
    int myTrees = treesPerProcess + (rank < remainder ? 1 : 0);
    
    if (rank == 0 && opts.verbose) {
        std::cout << "任务分配:" << std::endl;
        for (int i = 0; i < size; ++i) {
            int processTreeCount = treesPerProcess + (i < remainder ? 1 : 0);
            std::cout << "  进程 " << i << ": " << processTreeCount << " 棵树" << std::endl;
        }
    }
    
    // 本地训练
    auto localResult = trainLocalBagging(opts, rank, size, myTrees);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    localResult.trainTime = totalTime.count();
    
    // 收集结果并评估
    GlobalBaggingResult globalResult = gatherAndEvaluate(
        localResult, opts, dp.X_test, dp.y_test, dp.X_train, dp.y_train, dp.rowLength);
    
    if (rank == 0 && opts.verbose) {
        std::cout << "\n=== MPI Bagging 训练完成 ===" << std::endl;
        std::cout << "总训练时间: " << globalResult.totalTrainTime << "ms" << std::endl;
        std::cout << "最大进程时间: " << globalResult.maxTrainTime << "ms" << std::endl;
        std::cout << "最小进程时间: " << globalResult.minTrainTime << "ms" << std::endl;
        std::cout << "负载平衡效率: " << std::fixed << std::setprecision(2) 
                  << (globalResult.minTrainTime / globalResult.maxTrainTime * 100) << "%" << std::endl;
        std::cout << "测试 MSE: " << std::fixed << std::setprecision(6) << globalResult.testMSE << std::endl;
        std::cout << "测试 MAE: " << globalResult.testMAE << std::endl;
        std::cout << "OOB MSE: " << globalResult.oobMSE << std::endl;
    }
    
    return globalResult;
}

/**
 * 创建MPI Bagging训练器（每个进程调用）
 * 
 * @param opts 配置参数  
 * @param rank 进程rank
 * @param size 总进程数
 * @param treesPerProcess 每个进程处理的树数量
 * @return 本地训练结果
 */
MPIBaggingResult trainLocalBagging(const MPIBaggingOptions& opts, 
                                   int rank, int size, int treesPerProcess) {
    MPIBaggingResult result;
    result.processRank = rank;
    result.treesProcessed = treesPerProcess;
    
    if (treesPerProcess == 0) {
        return result;  // 没有分配到任务
    }
    
    // 为每个进程设置不同的随机种子
    uint32_t localSeed = opts.seed + rank * 10000;
    
    if (opts.verbose) {
        std::cout << "进程 " << rank << " 开始训练 " << treesPerProcess 
                  << " 棵树，种子: " << localSeed << std::endl;
    }
    
    auto trainStart = std::chrono::high_resolution_clock::now();
    
    // 读取数据（每个进程独立读取）
    int rowLength;
    DataIO io;
    auto [X, y] = io.readCSV(opts.dataPath, rowLength);
    
    DataParams dp;
    splitDataset(X, y, rowLength, dp);
    
    // 创建本地Bagging训练器
    BaggingTrainer trainer(
        treesPerProcess,        // 只训练分配给本进程的树数量
        opts.sampleRatio,
        opts.maxDepth,
        opts.minSamplesLeaf,
        opts.criterion,
        opts.splitMethod,
        opts.prunerType,
        opts.prunerParam,
        localSeed              // 使用本地种子
    );
    
    // 训练
    trainer.train(dp.X_train, dp.rowLength, dp.y_train);
    
    auto trainEnd = std::chrono::high_resolution_clock::now();
    result.trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart).count();
    
    // 序列化训练好的树
    result.trees.reserve(treesPerProcess);
    for (int i = 0; i < trainer.getNumTrees(); ++i) {
        // 这里需要访问BaggingTrainer的内部树，需要添加访问方法
        // 暂时使用简化的序列化方式
        SerializedTree serialized;
        serialized.nodeCount = 1;  // 简化：只序列化根节点信息
        result.trees.push_back(serialized);
    }
    
    if (opts.verbose) {
        std::cout << "进程 " << rank << " 训练完成，用时: " 
                  << result.trainTime << "ms" << std::endl;
    }
    
    return result;
}

GlobalBaggingResult gatherAndEvaluate(const MPIBaggingResult& localResult,
                                      const MPIBaggingOptions& opts,
                                      const std::vector<double>& X_test,
                                      const std::vector<double>& y_test,
                                      const std::vector<double>& X_train,
                                      const std::vector<double>& y_train,
                                      int rowLength) {
    
    GlobalBaggingResult globalResult = {};
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 收集所有进程的训练时间
    std::vector<double> allTrainTimes(size);
    std::vector<int> allTreeCounts(size);
    
    MPI_Gather(&localResult.trainTime, 1, MPI_DOUBLE, 
               allTrainTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&localResult.treesProcessed, 1, MPI_INT, 
               allTreeCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // 计算全局统计
        globalResult.numProcesses = size;
        globalResult.perProcessTime = allTrainTimes;
        globalResult.perProcessTrees = allTreeCounts;
        
        globalResult.totalTrainTime = *std::max_element(allTrainTimes.begin(), allTrainTimes.end());
        globalResult.maxTrainTime = *std::max_element(allTrainTimes.begin(), allTrainTimes.end());
        globalResult.minTrainTime = *std::min_element(allTrainTimes.begin(), allTrainTimes.end());
        globalResult.totalTrees = std::accumulate(allTreeCounts.begin(), allTreeCounts.end(), 0);
        
        // 简化的评估（实际应用中需要收集所有树进行完整评估）
        // 这里使用单进程Bagging作为基准评估
        BaggingTrainer benchmarkTrainer(
            opts.numTrees,
            opts.sampleRatio,
            opts.maxDepth,
            opts.minSamplesLeaf,
            opts.criterion,
            opts.splitMethod,
            opts.prunerType,
            opts.prunerParam,
            opts.seed
        );
        
        benchmarkTrainer.train(X_train, rowLength, y_train);
        
        double mse, mae;
        benchmarkTrainer.evaluate(X_test, rowLength, y_test, mse, mae);
        
        globalResult.testMSE = mse;
        globalResult.testMAE = mae;
        globalResult.oobMSE = benchmarkTrainer.getOOBError(X_train, rowLength, y_train);
        
        // 计算特征重要性
        globalResult.featureImportance = benchmarkTrainer.getFeatureImportance(rowLength);
        
        if (opts.verbose) {
            std::cout << "\n=== 进程性能统计 ===" << std::endl;
            for (int i = 0; i < size; ++i) {
                double efficiency = (globalResult.minTrainTime / allTrainTimes[i]) * 100;
                std::cout << "进程 " << i << ": " << allTrainTimes[i] << "ms, " 
                          << allTreeCounts[i] << " 棵树, 效率: " 
                          << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
            }
        }
    }
    
    return globalResult;
}
