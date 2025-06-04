// =============================================================================
// src/lightgbm/feature/FeatureBundler.cpp - 优化版本（避免vector<vector>）
// =============================================================================
#include "lightgbm/feature/FeatureBundler.hpp"
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <memory>

// 优化的冲突矩阵 - 使用一维数组模拟二维矩阵
class OptimizedConflictMatrix {
private:
    std::vector<double> data_;
    int size_;
    
public:
    explicit OptimizedConflictMatrix(int size) : size_(size) {
        data_.resize(size * size, 0.0);
    }
    
    double& operator()(int i, int j) {
        return data_[i * size_ + j];
    }
    
    const double& operator()(int i, int j) const {
        return data_[i * size_ + j];
    }
    
    int size() const { return size_; }
};

void FeatureBundler::createBundles(const std::vector<double>& data,
                                  int rowLength,
                                  size_t sampleSize,
                                  std::vector<FeatureBundle>& bundles) const {
    bundles.clear();
    
    // **优化1: 快速稀疏性检测**
    std::vector<double> sparsity(rowLength);
    constexpr double EPS = 1e-12;
    
    // **并行计算稀疏性**
    #pragma omp parallel for schedule(static) if(rowLength > 20)
    for (int f = 0; f < rowLength; ++f) {
        int nonZeroCount = 0;
        const size_t checkSize = std::min(sampleSize, size_t(5000)); // 采样检查，减少计算量
        
        for (size_t i = 0; i < checkSize; ++i) {
            if (std::abs(data[i * rowLength + f]) > EPS) {
                ++nonZeroCount;
            }
        }
        sparsity[f] = 1.0 - static_cast<double>(nonZeroCount) / checkSize;
    }
    
    // **优化2: 只对稀疏特征进行绑定**
    std::vector<int> sparseFeatures;
    std::vector<int> denseFeatures;
    constexpr double SPARSITY_THRESHOLD = 0.8; // 80%稀疏度阈值
    
    sparseFeatures.reserve(rowLength);
    denseFeatures.reserve(rowLength);
    
    for (int f = 0; f < rowLength; ++f) {
        if (sparsity[f] > SPARSITY_THRESHOLD) {
            sparseFeatures.push_back(f);
        } else {
            denseFeatures.push_back(f);
        }
    }
    
    // 密集特征直接单独成束
    bundles.reserve(denseFeatures.size() + sparseFeatures.size() / 2);
    for (int f : denseFeatures) {
        FeatureBundle bundle;
        bundle.features.push_back(f);
        bundle.offsets.push_back(0.0);
        bundle.totalBins = maxBin_;
        bundles.push_back(std::move(bundle));
    }
    
    if (sparseFeatures.size() < 2) {
        // 稀疏特征太少，直接单独处理
        for (int f : sparseFeatures) {
            FeatureBundle bundle;
            bundle.features.push_back(f);
            bundle.offsets.push_back(0.0);
            bundle.totalBins = maxBin_;
            bundles.push_back(std::move(bundle));
        }
        return;
    }
    
    // **优化3: 使用优化的冲突矩阵（一维数组模拟二维）**
    const int numSparse = static_cast<int>(sparseFeatures.size());
    OptimizedConflictMatrix conflictMatrix(numSparse);
    
    // **并行计算冲突矩阵**
    #pragma omp parallel for schedule(dynamic) if(numSparse > 10)
    for (int i = 0; i < numSparse; ++i) {
        for (int j = i + 1; j < numSparse; ++j) {
            const double conflict = calculateConflictRateOptimized(
                data, rowLength, sampleSize, sparseFeatures[i], sparseFeatures[j]);
            conflictMatrix(i, j) = conflictMatrix(j, i) = conflict;
        }
    }
    
    // **优化4: 改进的贪心绑定算法**
    std::vector<bool> used(numSparse, false);
    
    // 按稀疏度排序（更稀疏的优先）
    std::vector<std::pair<double, int>> sparsityWithIndex;
    sparsityWithIndex.reserve(numSparse);
    for (int i = 0; i < numSparse; ++i) {
        sparsityWithIndex.emplace_back(sparsity[sparseFeatures[i]], i);
    }
    std::sort(sparsityWithIndex.begin(), sparsityWithIndex.end(), std::greater<>());
    
    for (const auto& [sp, i] : sparsityWithIndex) {
        if (used[i]) continue;
        
        FeatureBundle bundle;
        bundle.features.push_back(sparseFeatures[i]);
        bundle.offsets.push_back(0.0);
        used[i] = true;
        
        double currentOffset = maxBin_;
        
        // 贪心添加兼容特征
        for (const auto& [sp2, j] : sparsityWithIndex) {
            if (used[j] || conflictMatrix(i, j) > maxConflictRate_) continue;
            
            // 检查与bundle中所有特征的兼容性
            bool compatible = true;
            for (int bundledFeature : bundle.features) {
                const int bundledIdx = std::find(sparseFeatures.begin(), sparseFeatures.end(), bundledFeature) 
                                     - sparseFeatures.begin();
                if (conflictMatrix(j, bundledIdx) > maxConflictRate_) {
                    compatible = false;
                    break;
                }
            }
            
            if (compatible && currentOffset + maxBin_ <= 65536) {
                bundle.features.push_back(sparseFeatures[j]);
                bundle.offsets.push_back(currentOffset);
                used[j] = true;
                currentOffset += maxBin_;
            }
        }
        
        bundle.totalBins = static_cast<int>(currentOffset);
        bundles.push_back(std::move(bundle));
    }
}

// **优化的冲突率计算方法**
double FeatureBundler::calculateConflictRateOptimized(const std::vector<double>& data,
                                                     int rowLength, 
                                                     size_t sampleSize,
                                                     int feat1, 
                                                     int feat2) const {
    constexpr double EPS = 1e-12;
    size_t conflicts = 0;
    size_t validPairs = 0;
    
    // **优化: 采样计算，减少计算量**
    const size_t checkSize = std::min(sampleSize, size_t(2000)); // 采样检查
    const size_t step = std::max(size_t(1), sampleSize / checkSize);
    
    for (size_t i = 0; i < sampleSize; i += step) {
        const double val1 = data[i * rowLength + feat1];
        const double val2 = data[i * rowLength + feat2];
        
        const bool nonZero1 = std::abs(val1) > EPS;
        const bool nonZero2 = std::abs(val2) > EPS;
        
        if (nonZero1 || nonZero2) {
            ++validPairs;
            if (nonZero1 && nonZero2) {
                ++conflicts;
            }
        }
    }
    
    return validPairs > 0 ? static_cast<double>(conflicts) / validPairs : 0.0;
}

// **保留原方法但使用优化实现**
double FeatureBundler::calculateConflictRate(const std::vector<double>& data,
                                           int rowLength, size_t sampleSize,
                                           int feat1, int feat2) const {
    return calculateConflictRateOptimized(data, rowLength, sampleSize, feat1, feat2);
}

void FeatureBundler::buildConflictGraph(const std::vector<double>& data,
                                      int rowLength, size_t sampleSize,
                                      std::vector<std::vector<double>>& conflictMatrix) const {
    // **优化: 使用扁平化的冲突矩阵计算**
    const int numFeatures = rowLength;
    OptimizedConflictMatrix optimizedMatrix(numFeatures);
    
    // **并行计算**
    #pragma omp parallel for schedule(dynamic) if(numFeatures > 10)
    for (int i = 0; i < numFeatures; ++i) {
        for (int j = i + 1; j < numFeatures; ++j) {
            const double conflict = calculateConflictRateOptimized(data, rowLength, sampleSize, i, j);
            optimizedMatrix(i, j) = optimizedMatrix(j, i) = conflict;
        }
    }
    
    // 转换回原格式（为了兼容性）
    conflictMatrix.assign(numFeatures, std::vector<double>(numFeatures, 0.0));
    for (int i = 0; i < numFeatures; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            conflictMatrix[i][j] = optimizedMatrix(i, j);
        }
    }
}

std::pair<int, double> FeatureBundler::transformFeature(int originalFeature,
                                                       double value,
                                                       const std::vector<FeatureBundle>& bundles) const {
    // **优化: 使用预计算的查找表（如果频繁调用的话）**
    for (size_t bundleIdx = 0; bundleIdx < bundles.size(); ++bundleIdx) {
        const auto& bundle = bundles[bundleIdx];
        
        // **优化: 对于小的bundle使用线性搜索，大的使用二分搜索**
        if (bundle.features.size() <= 8) {
            // 线性搜索
            for (size_t pos = 0; pos < bundle.features.size(); ++pos) {
                if (bundle.features[pos] == originalFeature) {
                    return transformFeatureValue(value, bundle.offsets[pos]);
                }
            }
        } else {
            // 二分搜索（如果特征已排序）
            auto it = std::lower_bound(bundle.features.begin(), bundle.features.end(), originalFeature);
            if (it != bundle.features.end() && *it == originalFeature) {
                const size_t pos = std::distance(bundle.features.begin(), it);
                return transformFeatureValue(value, bundle.offsets[pos]);
            }
        }
    }
    
    // 如果没找到，返回原值（这种情况不应该发生）
    return {originalFeature, value};
}

// **新增辅助方法**
std::pair<int, double> FeatureBundler::transformFeatureValue(double value, double offset) const {
    constexpr double EPS = 1e-12;
    
    double transformedValue;
    if (std::abs(value) < EPS) {
        // 零值特殊处理
        transformedValue = offset;
    } else {
        // **优化: 更精确的值转换**
        // 使用更稳定的映射函数
        const int binIndex = static_cast<int>(std::abs(value) * maxBin_ / 1000.0) % maxBin_;
        transformedValue = offset + binIndex + 1; // +1避免与零值冲突
    }
    
    return {0, transformedValue}; // bundleIdx在上层确定
}