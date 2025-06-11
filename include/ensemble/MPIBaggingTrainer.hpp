#pragma once
#include "ensemble/BaggingTrainer.hpp"
#include <mpi.h>
#include <vector>
#include <memory>

/**
 * MPI-enabled Bagging trainer that distributes tree training across MPI processes
 * Each process uses OpenMP for local parallelism
 */
class MPIBaggingTrainer {
public:
    MPIBaggingTrainer(int numTrees,
                      double sampleRatio,
                      int maxDepth,
                      int minSamplesLeaf,
                      const std::string& criterion,
                      const std::string& splitMethod,
                      const std::string& prunerType,
                      double prunerParam,
                      uint32_t seed);
    
    ~MPIBaggingTrainer();
    
    // Main training method - handles MPI distribution
    // numFeatures: actual number of features (without label column)
    void train(const std::vector<double>& data,
               int numFeatures,
               const std::vector<double>& labels);
    
    // Single prediction - aggregates predictions from all trees
    // numFeatures: actual number of features (without label column)
    double predict(const double* sample, int numFeatures) const;
    
    // Batch prediction - more efficient for multiple predictions
    // numFeatures: actual number of features (without label column)
    void predictBatch(const std::vector<double>& X,
                      int numFeatures,
                      std::vector<double>& predictions) const;
    
    // Evaluation across all trees
    // numFeatures: actual number of features (without label column)
    void evaluate(const std::vector<double>& X,
                  int numFeatures,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae);
    
    // Get feature importance aggregated from all trees
    std::vector<double> getFeatureImportance(int numFeatures) const;
    
    // Get OOB error (only available on master process)
    // numFeatures: actual number of features (without label column)
    double getOOBError(const std::vector<double>& data,
                       int numFeatures,
                       const std::vector<double>& labels) const;

private:
    // MPI-specific members
    int mpiRank_;
    int mpiSize_;
    MPI_Comm comm_;
    
    // Configuration
    int numTrees_;
    double sampleRatio_;
    int maxDepth_;
    int minSamplesLeaf_;
    std::string criterion_;
    std::string splitMethod_;
    std::string prunerType_;
    double prunerParam_;
    uint32_t baseSeed_;
    
    // Local trees (subset assigned to this process)
    std::unique_ptr<BaggingTrainer> localBagging_;
    int localNumTrees_;
    int treeOffset_;
    
    // Tree assignment calculation
    std::pair<int, int> calculateTreeAssignment(int rank, int size, int totalTrees) const;
    
    // Collective operations
    void gatherPredictions(const double* localPred, double* globalPred) const;
    void gatherFeatureImportance(const std::vector<double>& localImportance,
                                 std::vector<double>& globalImportance) const;
};