// =============================================================================
// include/lightgbm/feature/FeatureBundler.hpp - 优化版本
// =============================================================================
#pragma once

#include <vector>
#include <unordered_set>

struct FeatureBundle {
    std::vector<int> features;        
    std::vector<double> offsets;      
    int totalBins;                    
};

class FeatureBundler {
public:
    explicit FeatureBundler(int maxBin = 255, double maxConflictRate = 0.0)
        : maxBin_(maxBin), maxConflictRate_(maxConflictRate) {}
    
    // **主要方法**
    void createBundles(const std::vector<double>& data,
                      int rowLength, 
                      size_t sampleSize,
                      std::vector<FeatureBundle>& bundles) const;
    
    std::pair<int, double> transformFeature(int originalFeature, 
                                           double value,
                                           const std::vector<FeatureBundle>& bundles) const;

    // **兼容性方法（保留旧接口）**
    double calculateConflictRate(const std::vector<double>& data,
                               int rowLength, size_t sampleSize,
                               int feat1, int feat2) const;
    
    void buildConflictGraph(const std::vector<double>& data,
                          int rowLength, size_t sampleSize,
                          std::vector<std::vector<double>>& conflictMatrix) const;

private:
    int maxBin_;
    double maxConflictRate_;
    
    // **优化的内部方法**
    double calculateConflictRateOptimized(const std::vector<double>& data,
                                        int rowLength, size_t sampleSize,
                                        int feat1, int feat2) const;
    
    std::pair<int, double> transformFeatureValue(double value, double offset) const;
};