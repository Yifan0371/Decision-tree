#pragma once

#include "../tree/ITreeTrainer.hpp"
#include "../tree/ISplitFinder.hpp"
#include "../tree/ISplitCriterion.hpp"
#include "../tree/IPruner.hpp"
#include "tree/trainer/SingleTreeTrainer.hpp"
#include <vector>
#include <memory>
#include <random>


class BaggingTrainer : public ITreeTrainer {
public:
    
    BaggingTrainer(int numTrees = 10,
                   double sampleRatio = 1.0,
                   int maxDepth = 800,
                   int minSamplesLeaf = 2,
                   const std::string& criterion = "mse",
                   const std::string& splitMethod = "exhaustive",
                   const std::string& prunerType = "none",
                   double prunerParam = 0.01,
                   uint32_t seed = 42);

    
    void train(const std::vector<double>& data,
               int rowLength,
               const std::vector<double>& labels) override;

    double predict(const double* sample,
                   int rowLength) const override;

    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae) override;

    
    int getNumTrees() const { return numTrees_; }
    double getSampleRatio() const { return sampleRatio_; }
    
    
    std::vector<double> getFeatureImportance(int numFeatures) const;

    
    double getOOBError(const std::vector<double>& data,
                       int rowLength,
                       const std::vector<double>& labels) const;

private:
    
    int numTrees_;
    double sampleRatio_;
    int maxDepth_;
    int minSamplesLeaf_;
    std::string criterion_;
    std::string splitMethod_;
    std::string prunerType_;
    double prunerParam_;
    
    
    mutable std::mt19937 gen_;
    
    
    std::vector<std::unique_ptr<SingleTreeTrainer>> trees_;
    std::vector<std::vector<int>> oobIndices_;  
    
    
    std::unique_ptr<ISplitFinder> createSplitFinder() const;
    std::unique_ptr<ISplitCriterion> createCriterion() const;
    std::unique_ptr<IPruner> createPruner(const std::vector<double>& X_val,
                                         int rowLength,
                                         const std::vector<double>& y_val) const;
    
    
    void bootstrapSample(int dataSize,
                        std::vector<int>& sampleIndices,
                        std::vector<int>& oobIndices) const;
    
    
    void extractSubsetOptimized(const std::vector<double>& originalData,
                               int rowLength,
                               const std::vector<double>& originalLabels,
                               const std::vector<int>& indices,
                               std::vector<double>& subData,
                               std::vector<double>& subLabels) const;
    
    void bootstrapSample(int dataSize,
                        std::vector<int>& sampleIndices,
                        std::vector<int>& oobIndices,
                        std::mt19937& localGen) const;  
};
