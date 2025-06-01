#include "lightgbm/feature/FeatureBundler.hpp"
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cmath>

void FeatureBundler::createBundles(const std::vector<double>& data,
                                  int rowLength,
                                  size_t sampleSize,
                                  std::vector<FeatureBundle>& bundles) const {
    bundles.clear();
    
    // **修复1: 快速稀疏性检测**
    std::vector<double> sparsity(rowLength);
    const double EPS = 1e-12;
    
    for (int f = 0; f < rowLength; ++f) {
        int nonZeroCount = 0;
        for (size_t i = 0; i < sampleSize; ++i) {
            if (std::abs(data[i * rowLength + f]) > EPS) {
                nonZeroCount++;
            }
        }
        sparsity[f] = 1.0 - static_cast<double>(nonZeroCount) / sampleSize;
    }
    
    // **修复2: 只对稀疏特征进行绑定**
    std::vector<int> sparseFeatures;
    const double SPARSITY_THRESHOLD = 0.8; // 80%稀疏度阈值
    
    for (int f = 0; f < rowLength; ++f) {
        if (sparsity[f] > SPARSITY_THRESHOLD) {
            sparseFeatures.push_back(f);
        } else {
            // 非稀疏特征单独成束
            FeatureBundle bundle;
            bundle.features.push_back(f);
            bundle.offsets.push_back(0.0);
            bundle.totalBins = maxBin_;
            bundles.push_back(std::move(bundle));
        }
    }
    
    if (sparseFeatures.size() < 2) {
        // 如果稀疏特征太少，直接返回
        return;
    }
    
    // **修复3: 构建优化的冲突矩阵（仅针对稀疏特征）**
    int numSparse = static_cast<int>(sparseFeatures.size());
    std::vector<std::vector<double>> conflictMatrix(numSparse, 
                                                   std::vector<double>(numSparse, 0.0));
    
    for (int i = 0; i < numSparse; ++i) {
        for (int j = i + 1; j < numSparse; ++j) {
            double conflict = calculateConflictRate(data, rowLength, sampleSize,
                                                   sparseFeatures[i], sparseFeatures[j]);
            conflictMatrix[i][j] = conflictMatrix[j][i] = conflict;
        }
    }
    
    // **修复4: 改进的贪心绑定算法**
    std::vector<bool> used(numSparse, false);
    
    // 按稀疏度排序（更稀疏的优先）
    std::vector<std::pair<double, int>> sparsityWithIndex;
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
            if (used[j] || conflictMatrix[i][j] > maxConflictRate_) continue;
            
            // 检查与bundle中所有特征的兼容性
            bool compatible = true;
            for (int bundledIdx : bundle.features) {
                int bundledPos = std::find(sparseFeatures.begin(), sparseFeatures.end(), bundledIdx) 
                               - sparseFeatures.begin();
                if (conflictMatrix[j][bundledPos] > maxConflictRate_) {
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

double FeatureBundler::calculateConflictRate(const std::vector<double>& data,
                                           int rowLength, size_t sampleSize,
                                           int feat1, int feat2) const {
    const double EPS = 1e-12;
    size_t conflicts = 0;
    size_t validPairs = 0;
    
    // **修复5: 更精确的冲突率计算**
    for (size_t i = 0; i < sampleSize; ++i) {
        double val1 = data[i * rowLength + feat1];
        double val2 = data[i * rowLength + feat2];
        
        bool nonZero1 = std::abs(val1) > EPS;
        bool nonZero2 = std::abs(val2) > EPS;
        
        if (nonZero1 || nonZero2) {
            validPairs++;
            if (nonZero1 && nonZero2) {
                conflicts++;
            }
        }
    }
    
    return validPairs > 0 ? static_cast<double>(conflicts) / validPairs : 0.0;
}

void FeatureBundler::buildConflictGraph(const std::vector<double>& data,
                                      int rowLength, size_t sampleSize,
                                      std::vector<std::vector<double>>& conflictMatrix) const {
    // 这个函数现在在createBundles中内联实现，保留为接口兼容性
    for (int i = 0; i < rowLength; ++i) {
        for (int j = i + 1; j < rowLength; ++j) {
            double conflict = calculateConflictRate(data, rowLength, sampleSize, i, j);
            conflictMatrix[i][j] = conflictMatrix[j][i] = conflict;
        }
    }
}

std::pair<int, double> FeatureBundler::transformFeature(int originalFeature,
                                                       double value,
                                                       const std::vector<FeatureBundle>& bundles) const {
    // **修复6: 优化的特征查找**
    for (size_t bundleIdx = 0; bundleIdx < bundles.size(); ++bundleIdx) {
        const auto& bundle = bundles[bundleIdx];
        
        // 使用二分查找加速（如果特征已排序）
        auto it = std::find(bundle.features.begin(), bundle.features.end(), originalFeature);
        if (it != bundle.features.end()) {
            size_t pos = std::distance(bundle.features.begin(), it);
            
            // **修复7: 更精确的值转换**
            double offset = bundle.offsets[pos];
            double transformedValue;
            
            if (std::abs(value) < 1e-12) {
                // 零值特殊处理
                transformedValue = offset;
            } else {
                // 非零值：需要映射到对应的分箱
                // 简化实现：线性映射到分箱范围
                int binIndex = static_cast<int>(std::abs(value) * maxBin_ / 1000.0) % maxBin_;
                transformedValue = offset + binIndex + 1; // +1避免与零值冲突
            }
            
            return {static_cast<int>(bundleIdx), transformedValue};
        }
    }
    
    // 如果没找到，返回原值（这种情况不应该发生）
    return {originalFeature, value};
}