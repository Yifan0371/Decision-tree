#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <memory>

struct MPIBaggingOptions {
    std::string dataPath;        
    int         numTrees;        
    double      sampleRatio;     
    int         maxDepth;        
    int         minSamplesLeaf;  
    std::string criterion;       
    std::string splitMethod;     
    std::string prunerType;      
    double      prunerParam;     
    uint32_t    seed;
    
    // MPI特有参数
    bool        verbose;         // 是否输出详细信息
    bool        gatherResults;   // 是否收集所有结果到主进程
};

// 树序列化结构（用于MPI传输）
struct SerializedTree {
    std::vector<bool> isLeaf;           // 节点是否为叶子
    std::vector<int> featureIndex;      // 分裂特征索引
    std::vector<double> threshold;      // 分裂阈值
    std::vector<double> prediction;     // 叶子节点预测值
    std::vector<int> leftChild;         // 左子节点索引
    std::vector<int> rightChild;        // 右子节点索引
    std::vector<size_t> samples;        // 节点样本数
    std::vector<double> metric;         // 节点度量值
    int nodeCount;                      // 总节点数
    
    SerializedTree() : nodeCount(0) {}
};

// MPI通信用的结果结构
struct MPIBaggingResult {
    std::vector<SerializedTree> trees;  // 序列化的树
    std::vector<std::vector<int>> oobIndices; // OOB索引
    double trainTime;                   // 训练时间(ms)
    int processRank;                    // 进程rank
    int treesProcessed;                 // 处理的树数量
};

// 全局结果结构
struct GlobalBaggingResult {
    double testMSE;
    double testMAE; 
    double oobMSE;
    double totalTrainTime;
    double maxTrainTime;
    double minTrainTime;
    int totalTrees;
    std::vector<double> featureImportance;
    
    // MPI统计信息
    int numProcesses;
    std::vector<double> perProcessTime;
    std::vector<int> perProcessTrees;
};

/**
 * MPI+OpenMP混合并行Bagging应用主函数
 * 
 * @param opts 配置参数
 * @return 全局结果结构
 */
GlobalBaggingResult runMPIBaggingApp(const MPIBaggingOptions& opts);

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
                                   int rank, int size, int treesPerProcess);

/**
 * 序列化决策树以便MPI传输
 */
SerializedTree serializeTree(const class SingleTreeTrainer* tree);

/**
 * 反序列化决策树
 */
std::unique_ptr<class SingleTreeTrainer> deserializeTree(const SerializedTree& serialized);

/**
 * 收集所有进程的结果到主进程
 */
GlobalBaggingResult gatherAndEvaluate(const MPIBaggingResult& localResult,
                                      const MPIBaggingOptions& opts,
                                      const std::vector<double>& X_test,
                                      const std::vector<double>& y_test,
                                      const std::vector<double>& X_train,
                                      const std::vector<double>& y_train,
                                      int rowLength);