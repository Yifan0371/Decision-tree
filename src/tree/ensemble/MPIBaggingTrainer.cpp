#include "ensemble/MPIBaggingTrainer.hpp"
#include <iostream>
#include <iomanip>
#include <set>
#include <chrono>
#include <numeric>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

MPIBaggingTrainer::MPIBaggingTrainer(int numTrees,
                                     double sampleRatio,
                                     int maxDepth,
                                     int minSamplesLeaf,
                                     const std::string& criterion,
                                     const std::string& splitMethod,
                                     const std::string& prunerType,
                                     double prunerParam,
                                     uint32_t seed)
    : numTrees_(numTrees),
      sampleRatio_(sampleRatio),
      maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf),
      criterion_(criterion),
      splitMethod_(splitMethod),
      prunerType_(prunerType),
      prunerParam_(prunerParam),
      baseSeed_(seed),
      comm_(MPI_COMM_WORLD) {
    
    MPI_Comm_rank(comm_, &mpiRank_);
    MPI_Comm_size(comm_, &mpiSize_);
    
    // Calculate tree assignment for this process
    auto [localTrees, offset] = calculateTreeAssignment(mpiRank_, mpiSize_, numTrees_);
    localNumTrees_ = localTrees;
    treeOffset_ = offset;
    
    // Strong randomization for process independence
    std::random_device rd;
    std::mt19937 masterGen(seed);
    
    // Generate independent seed sequence
    std::vector<uint32_t> processSeeds(mpiSize_);
    for (int i = 0; i < mpiSize_; ++i) {
        processSeeds[i] = masterGen();
        processSeeds[i] ^= (i * 2654435761U);  // Large prime
        processSeeds[i] ^= rd();  // Hardware randomness
    }
    
    uint32_t localSeed = processSeeds[mpiRank_];
    uint32_t bagSeed = localSeed ^ (treeOffset_ * 1103515245U);
    
    // Create local bagging trainer
    localBagging_ = std::make_unique<BaggingTrainer>(
        localNumTrees_,
        sampleRatio_,
        maxDepth_,
        minSamplesLeaf_,
        criterion_,
        splitMethod_,
        prunerType_,
        prunerParam_,
        bagSeed
    );
    
    if (mpiRank_ == 0) {
        std::cout << "Enhanced MPI Bagging initialized with " << mpiSize_ << " processes" << std::endl;
        std::cout << "Total trees: " << numTrees_ << std::endl;
        std::cout << "Strong randomization: enabled" << std::endl;
        #ifdef _OPENMP
        std::cout << "OpenMP threads per process: " << omp_get_max_threads() << std::endl;
        #endif
    }
}

void MPIBaggingTrainer::train(const std::vector<double>& data,
                              int numFeatures,
                              const std::vector<double>& labels) {
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    if (mpiRank_ == 0) {
        std::cout << "\nStarting distributed training..." << std::endl;
        for (int r = 0; r < mpiSize_; ++r) {
            auto [trees, offset] = calculateTreeAssignment(r, mpiSize_, numTrees_);
            std::cout << "  Process " << r << ": trees " << offset 
                      << "-" << (offset + trees - 1) << " (" << trees << " trees)" << std::endl;
        }
    }
    
    auto trainStart = std::chrono::high_resolution_clock::now();
    
    // Data validation
    if (labels.empty() || data.empty() || numFeatures <= 0) {
        std::cerr << "Process " << mpiRank_ << " ERROR: Invalid training data!" << std::endl;
        return;
    }
    
    size_t expectedDataSize = labels.size() * numFeatures;
    if (data.size() != expectedDataSize) {
        std::cerr << "Process " << mpiRank_ << " ERROR: Data size mismatch!" << std::endl;
        return;
    }
    
    if (localNumTrees_ > 0) {
        localBagging_->train(data, numFeatures, labels);
    }
    
    auto trainEnd = std::chrono::high_resolution_clock::now();
    
    MPI_Barrier(comm_);
    auto totalEnd = std::chrono::high_resolution_clock::now();
    
    // Timing information
    auto localTrainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart).count();
    long maxTrainTime;
    MPI_Reduce(&localTrainTime, &maxTrainTime, 1, MPI_LONG, MPI_MAX, 0, comm_);
    
    if (mpiRank_ == 0) {
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
        std::cout << "\nMPI Bagging training completed!" << std::endl;
        std::cout << "Max training time across processes: " << maxTrainTime << "ms" << std::endl;
        std::cout << "Total time (including communication): " << totalTime.count() << "ms" << std::endl;
    }
}

double MPIBaggingTrainer::predict(const double* sample, int numFeatures) const {
    double localPred = 0.0;
    
    if (localNumTrees_ > 0 && localBagging_) {
        double avgPred = localBagging_->predict(sample, numFeatures);
        localPred = avgPred * localNumTrees_;
    }
    
    double globalSum = 0.0;
    MPI_Allreduce(&localPred, &globalSum, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
    return globalSum / numTrees_;
}

void MPIBaggingTrainer::predictBatch(const std::vector<double>& X,
                                     int numFeatures,
                                     std::vector<double>& predictions) const {
    
    if (X.size() % numFeatures != 0) {
        std::cerr << "Process " << mpiRank_ << " ERROR: X.size() not divisible by numFeatures" << std::endl;
        return;
    }
    
    const size_t n = X.size() / numFeatures;
    predictions.resize(n);
    
    std::vector<double> localPredictions(n, 0.0);
    
    if (localNumTrees_ > 0) {
        #pragma omp parallel for schedule(static, 256) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            localPredictions[i] = localBagging_->predict(&X[i * numFeatures], numFeatures) * localNumTrees_;
        }
    }
    
    // Ensure consistent batch sizes across processes
    int localSize = static_cast<int>(n);
    int globalSize = 0;
    MPI_Allreduce(&localSize, &globalSize, 1, MPI_INT, MPI_MAX, comm_);
    
    if (localSize != globalSize) {
        std::cerr << "Process " << mpiRank_ << " ERROR: Inconsistent batch sizes!" << std::endl;
        return;
    }
    
    MPI_Allreduce(localPredictions.data(), predictions.data(), 
                  static_cast<int>(n), MPI_DOUBLE, MPI_SUM, comm_);
    
    const double invNumTrees = 1.0 / numTrees_;
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] *= invNumTrees;
    }
}

void MPIBaggingTrainer::evaluate(const std::vector<double>& X,
                                 int numFeatures,
                                 const std::vector<double>& y,
                                 double& mse,
                                 double& mae) {
    
    const size_t n = y.size();
    
    // Data validation
    if (X.size() != n * numFeatures) {
        std::cerr << "Process " << mpiRank_ << " ERROR: Data size mismatch in evaluate!" << std::endl;
        mse = mae = std::numeric_limits<double>::infinity();
        return;
    }
    
    // Use batch prediction to avoid MPI communication in OpenMP loops
    std::vector<double> predictions;
    predictBatch(X, numFeatures, predictions);
    
    if (predictions.size() != n) {
        std::cerr << "Process " << mpiRank_ << " ERROR: Prediction size mismatch!" << std::endl;
        mse = mae = std::numeric_limits<double>::infinity();
        return;
    }
    
    // Compute MSE and MAE using OpenMP
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
    
    // Only master reports results
    if (mpiRank_ == 0) {
        std::cout << "Test MSE: " << std::fixed << std::setprecision(6) << mse 
                  << " | Test MAE: " << mae << std::endl;
    }
}

std::vector<double> MPIBaggingTrainer::getFeatureImportance(int numFeatures) const {
    // Ensure all processes use the same number of features
    int globalNumFeatures = numFeatures;
    MPI_Bcast(&globalNumFeatures, 1, MPI_INT, 0, comm_);
    
    // Calculate local importance
    std::vector<double> localImportance(globalNumFeatures, 0.0);
    
    if (localNumTrees_ > 0 && localBagging_) {
        try {
            auto tempImportance = localBagging_->getFeatureImportance(globalNumFeatures);
            
            if (tempImportance.size() == static_cast<size_t>(globalNumFeatures)) {
                // Scale by number of trees
                for (size_t i = 0; i < tempImportance.size(); ++i) {
                    localImportance[i] = tempImportance[i] * localNumTrees_;
                }
            } else {
                // Fallback: uniform importance
                double uniformImportance = 1.0 / globalNumFeatures;
                for (int i = 0; i < globalNumFeatures; ++i) {
                    localImportance[i] = uniformImportance * localNumTrees_;
                }
            }
        } catch (const std::exception&) {
            // Fallback: uniform importance
            double uniformImportance = 1.0 / globalNumFeatures;
            for (int i = 0; i < globalNumFeatures; ++i) {
                localImportance[i] = uniformImportance * localNumTrees_;
            }
        }
    }
    
    // Synchronization barrier
    MPI_Barrier(comm_);
    
    // Sum across all processes
    std::vector<double> globalImportance(globalNumFeatures, 0.0);
    int mpiResult = MPI_Allreduce(localImportance.data(), globalImportance.data(), 
                                  globalNumFeatures, MPI_DOUBLE, MPI_SUM, comm_);
    
    if (mpiResult != MPI_SUCCESS) {
        if (mpiRank_ == 0) {
            std::cerr << "Error: MPI_Allreduce failed in getFeatureImportance" << std::endl;
        }
        return std::vector<double>(globalNumFeatures, 0.0);
    }
    
    // Normalize by total trees (only on master process)
    if (mpiRank_ == 0 && numTrees_ > 0) {
        const double invNumTrees = 1.0 / numTrees_;
        for (auto& val : globalImportance) {
            val *= invNumTrees;
        }
    }
    
    // Broadcast the normalized result to all processes
    MPI_Bcast(globalImportance.data(), globalNumFeatures, MPI_DOUBLE, 0, comm_);
    
    return globalImportance;
}

double MPIBaggingTrainer::getOOBError(const std::vector<double>& data,
                                      int numFeatures,
                                      const std::vector<double>& labels) const {
    
    // OOB error calculation is complex in distributed setting
    // For now, return local OOB error on master process
    if (mpiRank_ == 0 && localNumTrees_ > 0) {
        return localBagging_->getOOBError(data, numFeatures, labels);
    }
    
    return 0.0;
}

MPIBaggingTrainer::~MPIBaggingTrainer() {
    // MPI cleanup handled by main program
}

std::pair<int, int> MPIBaggingTrainer::calculateTreeAssignment(int rank, int size, int totalTrees) const {
    // Distribute trees as evenly as possible
    int baseTreesPerProcess = totalTrees / size;
    int remainder = totalTrees % size;
    
    int localTrees = baseTreesPerProcess;
    int offset = rank * baseTreesPerProcess;
    
    // Distribute remainder trees to first 'remainder' processes
    if (rank < remainder) {
        localTrees++;
        offset += rank;
    } else {
        offset += remainder;
    }
    
    return {localTrees, offset};
}

void MPIBaggingTrainer::gatherPredictions(const double* localPred, double* globalPred) const {
    MPI_Allreduce(localPred, globalPred, 1, MPI_DOUBLE, MPI_SUM, comm_);
    *globalPred /= numTrees_;
}

void MPIBaggingTrainer::gatherFeatureImportance(const std::vector<double>& localImportance,
                                                 std::vector<double>& globalImportance) const {
    int numFeatures = static_cast<int>(localImportance.size());
    globalImportance.resize(numFeatures);
    
    // Scale local importance by number of trees
    std::vector<double> scaledLocal = localImportance;
    for (auto& val : scaledLocal) {
        val *= localNumTrees_;
    }
    
    MPI_Allreduce(scaledLocal.data(), globalImportance.data(), 
                  numFeatures, MPI_DOUBLE, MPI_SUM, comm_);
    
    // Normalize
    for (auto& val : globalImportance) {
        val /= numTrees_;
    }
}