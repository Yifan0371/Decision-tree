#include "ensemble/MPIBaggingTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <sstream>

struct MPIBaggingOptions {
    std::string dataPath;
    int numTrees;
    double sampleRatio;
    int maxDepth;
    int minSamplesLeaf;
    std::string criterion;
    std::string splitMethod;
    std::string prunerType;
    double prunerParam;
    uint32_t seed;
};

void printUsage(const char* programName) {
    std::cout << "Usage: mpirun -np <num_processes> " << programName << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  <dataPath>       - Path to CSV data file" << std::endl;
    std::cout << "  <numTrees>       - Total number of trees (distributed across processes)" << std::endl;
    std::cout << "  <sampleRatio>    - Bootstrap sample ratio (default: 1.0)" << std::endl;
    std::cout << "  <maxDepth>       - Maximum tree depth (default: 800)" << std::endl;
    std::cout << "  <minSamplesLeaf> - Minimum samples per leaf (default: 2)" << std::endl;
    std::cout << "  <criterion>      - Split criterion (default: mse)" << std::endl;
    std::cout << "  <splitMethod>    - Split method (default: exhaustive)" << std::endl;
    std::cout << "  <prunerType>     - Pruner type (default: none)" << std::endl;
    std::cout << "  <prunerParam>    - Pruner parameter (default: 0.01)" << std::endl;
    std::cout << "  <seed>           - Random seed (default: 42)" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  mpirun -np 4 " << programName << " data.csv 100 1.0" << std::endl;
}

// 重定向cout以隐藏训练过程输出
class OutputRedirector {
private:
    std::streambuf* orig_cout_buf;
    std::ostringstream null_stream;

public:
    void silence() {
        orig_cout_buf = std::cout.rdbuf();
        std::cout.rdbuf(null_stream.rdbuf());
    }
    
    void restore() {
        std::cout.rdbuf(orig_cout_buf);
    }
};

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Set default parameters
    MPIBaggingOptions opts;
    opts.dataPath = "../data/data_clean/cleaned_data.csv";
    opts.numTrees = 100;
    opts.sampleRatio = 1.0;
    opts.maxDepth = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion = "mse";
    opts.splitMethod = "exhaustive";
    opts.prunerType = "none";
    opts.prunerParam = 0.01;
    opts.seed = 42;
    
    // Parse arguments
    if (argc < 2) {
        if (mpiRank == 0) {
            std::cout << "Warning: No arguments provided. Using defaults." << std::endl;
            printUsage(argv[0]);
        }
    } else {
        if (argc >= 2) opts.dataPath = argv[1];
        if (argc >= 3) opts.numTrees = std::stoi(argv[2]);
        if (argc >= 4) opts.sampleRatio = std::stod(argv[3]);
        if (argc >= 5) opts.maxDepth = std::stoi(argv[4]);
        if (argc >= 6) opts.minSamplesLeaf = std::stoi(argv[5]);
        if (argc >= 7) opts.criterion = argv[6];
        if (argc >= 8) opts.splitMethod = argv[7];
        if (argc >= 9) opts.prunerType = argv[8];
        if (argc >= 10) opts.prunerParam = std::stod(argv[9]);
        if (argc >= 11) opts.seed = static_cast<uint32_t>(std::stoi(argv[10]));
    }
    
    // Validate parameters
    if (opts.dataPath.empty()) {
        if (mpiRank == 0) {
            std::cerr << "Error: Data path is empty!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (opts.numTrees <= 0) {
        if (mpiRank == 0) {
            std::cerr << "Error: Number of trees must be positive!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Master process prints configuration
    if (mpiRank == 0) {
        std::cout << "=== MPI Bagging Parameters ===" << std::endl;
        std::cout << "MPI Processes: " << mpiSize << std::endl;
        std::cout << "Data: " << opts.dataPath << std::endl;
        std::cout << "Trees: " << opts.numTrees << " (distributed)" << std::endl;
        std::cout << "Sample Ratio: " << opts.sampleRatio << std::endl;
        std::cout << "Max Depth: " << opts.maxDepth << std::endl;
        std::cout << "Min Samples Leaf: " << opts.minSamplesLeaf << std::endl;
        std::cout << "Criterion: " << opts.criterion << std::endl;
        std::cout << "Split Method: " << opts.splitMethod << std::endl;
        std::cout << "Pruner: " << opts.prunerType;
        if (opts.prunerType != "none") {
            std::cout << " (" << opts.prunerParam << ")";
        }
        std::cout << std::endl;
        std::cout << "Seed: " << opts.seed << std::endl;
        std::cout << "==============================" << std::endl;
    }
    
    try {
        // Unified data dimension management
        std::vector<double> trainX, trainY;
        std::vector<double> testX, testY;
        int numFeatures = 0;  // Actual number of features (excluding label)
        
        // Only master process loads and splits data
        if (mpiRank == 0) {
            std::cout << "Loading data from: " << opts.dataPath << std::endl;
            
            DataIO io;
            int rawRowLength = 0;
            auto [X, y] = io.readCSV(opts.dataPath, rawRowLength);
            
            if (X.empty() || y.empty()) {
                std::cerr << "Error: Failed to load data from " << opts.dataPath << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Calculate actual number of features
            numFeatures = rawRowLength - 1;  // CSV rowLength includes label column
            
            // Split dataset
            DataParams dp;
            if (!splitDataset(X, y, rawRowLength, dp)) {
                std::cerr << "Failed to split dataset" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            std::cout << "Dataset loaded: " << y.size() << " samples, " 
                      << numFeatures << " features" << std::endl;
            std::cout << "Train: " << dp.y_train.size() << " samples" << std::endl;
            std::cout << "Test: " << dp.y_test.size() << " samples" << std::endl;
            
            trainX = std::move(dp.X_train);
            trainY = std::move(dp.y_train);
            testX = std::move(dp.X_test);
            testY = std::move(dp.y_test);
        }
        
        // Broadcast number of features to all processes
        MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Validate number of features
        if (numFeatures <= 0) {
            if (mpiRank == 0) {
                std::cerr << "Error: Invalid number of features: " << numFeatures << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Distribute training data
        int trainSize = 0;
        if (mpiRank == 0) {
            trainSize = static_cast<int>(trainY.size());
        }
        MPI_Bcast(&trainSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Allocate training data space
        if (mpiRank != 0) {
            trainX.resize(trainSize * numFeatures);
            trainY.resize(trainSize);
        }
        
        // Broadcast training data
        MPI_Bcast(trainX.data(), trainSize * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(trainY.data(), trainSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Create MPI Bagging trainer
        MPIBaggingTrainer trainer(
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
        
        // **关键修改**: 隐藏训练过程中的详细输出
        OutputRedirector redirector;
        if (mpiRank != 0) {
            // 非主进程完全静默
            redirector.silence();
        }
        
        // Training phase
        auto trainStart = std::chrono::high_resolution_clock::now();
        
        // 主进程在训练期间也临时静默，避免过多输出
        bool masterSilenced = false;
        if (mpiRank == 0) {
            redirector.silence();
            masterSilenced = true;
        }
        
        trainer.train(trainX, numFeatures, trainY);
        auto trainEnd = std::chrono::high_resolution_clock::now();
        
        // 恢复主进程输出
        if (masterSilenced) {
            redirector.restore();
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 计算和输出训练时间
        auto localTrainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart).count();
        long maxTrainTime;
        MPI_Reduce(&localTrainTime, &maxTrainTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (mpiRank == 0) {
            auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
            std::cout << "Max training time across processes: " << maxTrainTime << "ms" << std::endl;
            std::cout << "Total time (including communication): " << totalTime.count() << "ms" << std::endl;
        }
        
        // Distribute test data
        int testSize = 0;
        if (mpiRank == 0) {
            testSize = static_cast<int>(testY.size());
            std::cout << "Preparing test data: " << testSize << " samples with " 
                      << numFeatures << " features" << std::endl;
        }
        
        MPI_Bcast(&testSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (testSize <= 0) {
            if (mpiRank == 0) {
                std::cerr << "Error: No test data!" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Allocate test data space
        if (mpiRank != 0) {
            testX.resize(testSize * numFeatures);
            testY.resize(testSize);
        }
        
        // Broadcast test data
        MPI_Bcast(testX.data(), testSize * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(testY.data(), testSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Evaluation
        if (mpiRank == 0) {
            std::cout << "Evaluating model..." << std::endl;
        }
        
        double mse = 0.0, mae = 0.0;
        trainer.evaluate(testX, numFeatures, testY, mse, mae);
        
        // Feature importance calculation
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (mpiRank == 0) {
            std::cout << "Computing feature importance..." << std::endl;
        }
        
        try {
            // All processes must call getFeatureImportance due to MPI communication
            auto importance = trainer.getFeatureImportance(numFeatures);
            
            // Only master process outputs results
            if (mpiRank == 0) {
                std::cout << "Feature importance computed successfully." << std::endl;
                
                // Find top features
                std::vector<std::pair<double, int>> importanceWithIndex;
                for (int i = 0; i < static_cast<int>(importance.size()); ++i) {
                    importanceWithIndex.emplace_back(importance[i], i);
                }
                
                std::sort(importanceWithIndex.begin(), importanceWithIndex.end(),
                          std::greater<std::pair<double, int>>());
                
                std::cout << "\nTop 10 Feature Importances:" << std::endl;
                for (int i = 0; i < std::min(10, static_cast<int>(importanceWithIndex.size())); ++i) {
                    std::cout << "Feature " << importanceWithIndex[i].second
                              << ": " << std::fixed << std::setprecision(4)
                              << importanceWithIndex[i].first << std::endl;
                }
                
                // Timing summary
                auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
                std::cout << "\nTiming Summary:" << std::endl;
                std::cout << "Training time: " << trainTime.count() << "ms" << std::endl;
                std::cout << "Trees per process: ~" << opts.numTrees / mpiSize << std::endl;
                std::cout << "\n=== MPI+OpenMP Bagging Results ===" << std::endl;
                std::cout << "Final MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;
                std::cout << "Final MAE: " << std::fixed << std::setprecision(6) << mae << std::endl;
                std::cout << "Total Trees: " << opts.numTrees << " (distributed across " << mpiSize << " processes)" << std::endl;
                std::cout << "Features: " << numFeatures << std::endl;
                std::cout << "MPI+OpenMP Bagging completed successfully!" << std::endl;
            }
            
        } catch (const std::exception& e) {
            if (mpiRank == 0) {
                std::cout << "Warning: Feature importance calculation failed: " << e.what() << std::endl;
                std::cout << "But training and evaluation completed successfully!" << std::endl;
            }
        }
        
        // 恢复非主进程的输出
        if (mpiRank != 0) {
            redirector.restore();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    } catch (const std::exception& e) {
        std::cerr << "Process " << mpiRank << " error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}